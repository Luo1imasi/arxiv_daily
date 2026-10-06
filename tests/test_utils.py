import unittest

from arxiv_daily.utils import parallel_execute, retry_call


class ParallelExecuteTests(unittest.TestCase):
    def test_preserves_input_order(self):
        values = parallel_execute(lambda value: value * 2, [3, 1, 2], max_workers=3)

        self.assertEqual(values, [6, 2, 4])

    def test_can_raise_on_worker_error(self):
        def fail_on_two(value: int) -> int:
            if value == 2:
                raise ValueError("bad value")
            return value

        with self.assertRaisesRegex(RuntimeError, "parallel task"):
            parallel_execute(fail_on_two, [1, 2, 3], max_workers=3, raise_on_error=True)


class RetryCallTests(unittest.TestCase):
    def test_non_retryable_error_is_not_retried(self):
        calls = {"count": 0}

        class Denied(Exception):
            retryable = False

        def fail():
            calls["count"] += 1
            raise Denied("no")

        with self.assertRaises(Denied):
            retry_call(fail, max_retries=3, base_delay=0, exceptions=(Denied,))

        self.assertEqual(calls["count"], 1)

    def test_retry_after_is_preferred_over_backoff(self):
        delays: list[float] = []

        class Limited(Exception):
            retry_after = 2.5

        def fail():
            raise Limited("slow")

        old_sleep = __import__("arxiv_daily.utils", fromlist=["time"]).time.sleep
        import arxiv_daily.utils as utils_module

        utils_module.time.sleep = lambda delay: delays.append(delay)
        try:
            with self.assertRaises(Limited):
                retry_call(fail, max_retries=2, base_delay=30, max_delay=10, exceptions=(Limited,))
        finally:
            utils_module.time.sleep = old_sleep

        self.assertEqual(delays, [2.5])


if __name__ == "__main__":
    unittest.main()
