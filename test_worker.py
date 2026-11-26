import unittest
from pathlib import Path

from worker import block_for_task


class BlockDistributionTests(unittest.TestCase):
    def test_four_indices_cover_every_image_once(self) -> None:
        images = [Path(f"image-{index:02}.jpg") for index in range(11)]
        blocks = [block_for_task(images, index, 4) for index in range(4)]

        self.assertEqual([len(block) for block in blocks], [2, 3, 3, 3])
        self.assertEqual([path for block in blocks for path in block], images)

    def test_empty_work_is_valid_when_tasks_outnumber_images(self) -> None:
        images = [Path("only.jpg")]

        self.assertEqual(block_for_task(images, 0, 4), [])
        self.assertEqual(block_for_task(images, 3, 4), images)

    def test_rejects_an_index_outside_the_job(self) -> None:
        with self.assertRaises(ValueError):
            block_for_task([], 4, 4)


if __name__ == "__main__":
    unittest.main()
