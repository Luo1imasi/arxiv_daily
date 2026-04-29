import unittest

from arxiv_daily.utils import parallel_execute


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


if __name__ == "__main__":
    unittest.main()
