import unittest
from ray_retriever.utils.common_utils import partition, obfuscate_password

class CommonUtilsCases(unittest.TestCase):

    def test_partition(self):

        lst = [0,1,2,3,4,5,6,7,8,9]

        result = partition(lst, 1)
        self.assertEqual(list(result), [[0,1,2,3,4,5,6,7,8,9]])

        result = partition(lst, 2)
        self.assertEqual(list(result), [[0,1,2,3,4], [5,6,7,8,9]])

        result = partition(lst, 3)
        self.assertEqual(list(result), [[0,1,2,3], [4,5,6,7], [8,9]])

        result = partition(lst, 10)
        self.assertEqual(list(result), [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

        result = partition(lst, 99)
        self.assertEqual(list(result), [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

    def test_obfuscate_password(self):
        password = "0123456789"

        result = obfuscate_password(None, 4)
        self.assertEqual(result, None)

        result = obfuscate_password('', 4)
        self.assertEqual(result, '')

        result = obfuscate_password(password, 0)
        self.assertEqual(result, "**********")

        result = obfuscate_password(password, 4)
        self.assertEqual(result, "0123******")

        result = obfuscate_password(password, len(password))
        self.assertEqual(result, "0123456789")

        result = obfuscate_password(password, len(password)+1)
        self.assertEqual(result, "0123456789")