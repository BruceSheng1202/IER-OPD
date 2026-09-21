import asyncio
import atexit
import os
import queue
import threading
import time

# Import core functions from sglang_rollout directly to avoid code duplication
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        # When True, the worker loop stops taking new data and finishes any
        # in-flight groups, so weight updates can run without interrupting an
        # in-progress decode (which makes sglang return partial top-logprobs).
        self.paused = False
        self.active_tasks = set()
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if self.active_tasks:
                    done_tasks = {task for task in self.active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    self.active_tasks -= done_tasks

                # If paused (e.g. about to update weights), do not start any new
                # generation task this iteration. Let in-flight groups finish
                # naturally so the subsequent pause_generation / update_weights
                # does not interrupt an in-progress decode -- interrupting it
                # makes sglang return partial output_top_logprobs (shorter than
                # output_token_logprobs), corrupting student top-k.
                while not self.paused and len(self.active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                result = done_task.result()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        self.active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if self.active_tasks:
            print(f"Waiting for {len(self.active_tasks)} continuous tasks to complete...")
            await asyncio.wait(self.active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        # Use a thread-local selector event loop to avoid sharing libuv state
        # with Ray's other event loops.
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            print(f"ASYNC_WORKER_LOOP={type(runner.get_loop()).__module__}.{type(runner.get_loop()).__name__}", flush=True)
            runner.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def pause(self):
        """Stop taking new generation tasks. In-flight groups are allowed to
        finish. Call before a weight update so pause_generation/update_weights
        does not interrupt an in-progress decode (which would make sglang
        return partial top-logprobs)."""
        self.paused = True

    def resume(self):
        """Resume taking new generation tasks after a weight update."""
        self.paused = False

    def num_active_tasks(self) -> int:
        # Mirror the worker-loop cleanup: drop already-finished tasks from the
        # count so the caller sees the number still actually in flight.
        if self.active_tasks:
            done = {task for task in self.active_tasks if task.done()}
            for task in done:
                try:
                    task.result()
                except Exception as e:
                    print(f"Task failed with exception: {e}")
            self.active_tasks -= done
        return len(self.active_tasks)

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    # add back to buffer so it can be retried or handled by buffer policy
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# Persist drain/resume events because Ray can buffer or suppress worker stdout.
_DRAIN_LOG_PATH = os.environ.get(
    "ASYNC_DRAIN_LOG", os.path.join(os.environ.get("OUTPUT_ROOT", "outputs"), "logs", "async_drain.log")
)


def _log_drain(msg: str) -> None:
    try:
        with open(_DRAIN_LOG_PATH, "a") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            fh.flush()
    except Exception as e:
        print(f"[drain] failed to write drain log: {e}", flush=True)


def drain_for_weight_update(timeout: float = 120.0, poll_interval: float = 0.5) -> dict:
    """Pause the background worker and wait until every in-flight rollout group
    has finished decoding, so the caller can run pause_generation / update_weights
    without interrupting an in-progress decode.

    Returns a small report dict. The caller MUST call ``resume_after_weight_update``
    once the weight update is done, otherwise the worker stays paused forever.

    Soft-drain by design: we never abort in-flight requests. Aborting them would
    make sglang return partial output (tokens complete, top-logprobs truncated),
    which is exactly the corruption we are guarding against. On timeout we still
    proceed (the caller updates weights anyway) but report it so it is visible.
    """
    worker = _global_worker
    if worker is None:
        _log_drain("drain: no_worker (global worker not created yet)")
        return {"status": "no_worker", "active": 0}
    worker.pause()
    waited = 0.0
    last_active = worker.num_active_tasks()
    t0 = time.time()
    while last_active > 0 and waited < timeout:
        time.sleep(poll_interval)
        waited += poll_interval
        last_active = worker.num_active_tasks()
    status = "drained" if last_active == 0 else "timeout"
    report = {"status": status, "active": last_active, "waited_s": round(waited, 1)}
    _log_drain(
        f"drain: status={status} active={last_active} waited={waited:.1f}s "
        f"(entered with {worker.num_active_tasks()} after pause, t={time.time()-t0:.1f}s)"
    )
    return report


def resume_after_weight_update() -> None:
    """Resume the background worker after a weight update (counterpart of
    ``drain_for_weight_update``)."""
    worker = _global_worker
    if worker is not None:
        worker.resume()
        _log_drain("resume: worker un-paused")
    else:
        _log_drain("resume: no_worker")



# Register exit cleanup function

atexit.register(stop_global_worker)
