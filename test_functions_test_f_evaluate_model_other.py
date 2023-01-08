import json
import unittest
from datetime import datetime

from functions_f_evaluate_model_20221116 import update_results

class TestUpdate(unittest.TestCase):


    def test_failure(self):
        self.assertEqual(5, 6)

    def test_update(self):
        updated = {}
        update_results(updated, make_result(score=10, time=0.1, method="method"), key='TEST', directory='./offline/')
        is_ok = return_for_assert_ok(updated, 'offline/results_test_results/expected1.json', testname='test 1')
        self.assertEqual(is_ok, [])


def return_for_assert_ok(updated, expected__json_filename, testname=""):
    from deepdiff import DeepDiff

    with open(expected__json_filename) as f:
        expected1 = json.loads(f.read())

        answer = DeepDiff(expected1, updated)
        failed = []
        for key00, keys in answer.items():
            if key00 == 'values_changed':
                for key, value in keys.items():
                    if key not in ["root['TEST']['date']", "root['TEST']['first run']", "root['TEST']['best run date']", "root['TEST']['silver run date']"]:
                        failed.append(key)
            else:
                failed.append(key00)
        if failed != []:
            print(testname, ":", failed, 'should be empty')
            print()
            print("differences")
            print(answer)
            print()
            print(testname, ":", failed, 'should be empty')
            print("changed")
            print(answer['values_changed'])
            print("updated json")
            print(updated)
        #assert failed == []
        return failed

def make_result(score, time, method, vary=""):
    return {
        # "Mean Absolute Error Accuracy": score,
        # "Mean Squared Error Accuracy": score,
        # "R square Accuracy": score,
        # "Root Mean Squared Error": score,
        '_score': score,
        '_train time': time,
        # "best params": {
        #    "param1": "param1_" + str(score),
        #    "param2": "param2_" + str(score),
        #    "param3": "param3_" + str(score),
        # },
        "date": str(datetime.now()),
        "_method": method,
        # "first run": "2022-11-06 22:13:02.393884",
        '_params': {
            "param1": "param1_" + str(score) + vary,
            "param2": "param2_" + str(score),
            "param3": "param3_" + str(score),
        },
        "random_state": 101
    }

if __name__ == '__main__':
    unittest.main()