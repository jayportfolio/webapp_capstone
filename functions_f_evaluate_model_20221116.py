import pandas as pd

from time import time
from datetime import datetime
import json


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


def update_results(key, saved_results_json, new_results):

    raise EnvironmentError("this isn't the right update_results method")
    try:
        first_run_date = str(datetime.now())
        first_run_date = saved_results_json[key]['date']
        first_run_date = saved_results_json[key]['first run']
    except:
        pass

    try:
        best_score = -1000
        best_params = 'NOT APPLICABLE'
        best_time = 99999999
        best_score = saved_results_json[key]['Score']
        best_params = saved_results_json[key]['params']
        best_time = saved_results_json[key]['Training Time']
        best_score = saved_results_json[key]['best score']
        best_params = saved_results_json[key]['best params']
        best_time = saved_results_json[key]['best time']
    except:
        pass

    new_results['first run'] = first_run_date

    if key not in saved_results_json:
        new_results['best params'] = new_results['params']
        new_results['best score'] = new_results['Score']
        new_results['best time'] = new_results['Training Time']
        new_results['suboptimal'] = 'pending'

    elif best_score > saved_results_json[key]['Score']:
        new_results['suboptimal'] = 'suboptimal'

    elif best_score == saved_results_json[key]['Score']:
        if saved_results_json[key]['params'] != new_results['params']:
            new_results['best params'] = 'MULTIPLE PARAM OPTIONS'
            new_results['best is shared'] = True
            if new_results['Training Time'] < best_time:
                new_results['best params'] = new_results['params']
                new_results['best score'] = new_results['Score']
                new_results['best time'] = new_results['Training Time']
                new_results['suboptimal'] = 'pending'
            else:
                new_results['best params'] = saved_results_json[key]['params']
                new_results['best score'] = saved_results_json[key]['Score']
                new_results['best time'] = saved_results_json[key]['Training Time']
                new_results['suboptimal'] = 'pending'

        else:
            new_results['best params'] = saved_results_json[key]['params']
            new_results['best score'] = saved_results_json[key]['Score']
            new_results['suboptimal'] = 'pending'

    else:
        new_results['best params'] = saved_results_json[key]['params']
        new_results['best score'] = saved_results_json[key]['Score']
        new_results['suboptimal'] = 'pending'

    saved_results_json[key] = new_results

    return saved_results_json


def get_best_estimator_average_time(best_estimator_pipe, X_train, y_train, debug=False):
    timings = []

    max_time_iter = 5 if debug else 1

    for i in range(0, max_time_iter):
        t0 = time()
        best_estimator_pipe.fit(X_train, y_train)
        timings.append(time() - t0)
        if time() - t0 > 30: print(i, ":", time() - t0)

    print(timings)
    average_time = sum(timings) / len(timings)

    return average_time


def get_results(directory='../../../results/'):
    results_filename = directory+'results.json'

    with open(results_filename) as f:
        raw_audit = f.read()
    results_json = json.loads(raw_audit)
    return results_json


def update_results(saved_results_json, new_results, key, directory='../../../results/', aim='maximize'):
    bad = []
    for each in ['_params', '_score', '_train time', 'date', 'random_state']:
        if each not in list(new_results.keys()):
            bad.append(each)
        if len(bad) > 0:
            raise ValueError(str(bad) + ' should be in the results array')

    first_run_date = str(datetime.now())
    if saved_results_json is not None and key in saved_results_json:
        old_results = saved_results_json[key]

    try:
        first_run_date = old_results['date']
        first_run_date = old_results['first run']
    except:
        pass

    max_score = -1000
    try:
        max_score = max(max_score, old_results['_score'])
        max_score = max(max_score, old_results['best score'])
    except:
        pass

    new_results['first run'] = first_run_date

    new_results['best is shared'] = False

    if key not in saved_results_json:
        put_new_in_best(new_results, saved_results_json)
        this_model_is_best = True
    elif max_score > new_results['_score']:
        put_old_best_in_best(new_results, old_results)

        this_model_is_best = False
    elif max_score == new_results['_score']:

        if old_results['best params'] == new_results['_params'] and new_results['_train time'] * 1.3 <= old_results['best time']:
            print("3a: ? same params but better time ==> replace results and update model")

            #put_new_in_best(new_results, old_results)
            replace_new_in_best(new_results, old_results)
            this_model_is_best = True

        elif old_results['best params'] != new_results['_params'] and new_results['_train time'] < old_results['best time']:
            print("3b: ? different params and better time ==> update results and update model")

            put_new_in_best(new_results, old_results)
            new_results['best is shared'] = True

            this_model_is_best = True

        elif old_results['best params'] == new_results['_params'] or new_results['_train time'] * 3 < old_results['best time']:
            print("3c: ? same params or much better time ==> don't update results and don't update model")

            put_old_best_in_best(new_results, old_results)  ## was best2

            this_model_is_best = False

        else:
            print("3z: ? something else ==> share best results and don't update model")

            put_old_best_in_best(new_results, old_results)  ## was best2
            new_results['best is shared'] = True

            this_model_is_best = False

    else:
        put_new_in_best(new_results, old_results)

        this_model_is_best = True

    saved_results_json[key] = new_results.copy()

    results_filename = directory + 'results.json'
    with open(results_filename, 'w') as file:
        file.write(json.dumps(saved_results_json, indent=4, sort_keys=True))

    return this_model_is_best


def put_old_best_in_best(new_results, old_results):
    if 'silver params' not in old_results:
        new_results['silver score'] = new_results['_score']
        new_results['silver time'] = new_results['_train time']
        new_results['silver params'] = new_results['_params']
        new_results['silver method'] = new_results['_method']
        new_results['silver run date'] = new_results['date']
    elif new_results['_score'] > old_results['silver score']:
        new_results['silver score'] = new_results['_score']
        new_results['silver time'] = new_results['_train time']
        new_results['silver params'] = new_results['_params']
        new_results['silver method'] = new_results['_method']
        new_results['silver run date'] = new_results['date']
    else:
        new_results['silver score'] = old_results['silver score']
        new_results['silver time'] = old_results['silver time']
        new_results['silver params'] = old_results['silver params']
        new_results['silver method'] = old_results['silver method']
        new_results['silver run date'] = old_results['silver run date']

    new_results['best score'] = old_results['best score']
    new_results['best time'] = old_results['best time']
    new_results['best params'] = old_results['best params']
    new_results['best method'] = old_results['best method']
    new_results['best run date'] = old_results['best run date']

    new_results['suboptimal'] = 'suboptimal'


def put_new_in_best(new_results, old_results):
    # if 'best score' in new_results and ('silver params' not in new_results or new_results['best score'] > new_results['silver score']):
    if 'best params' in old_results and new_results['_params'] == old_results['best params']:
        if 'silver score' in old_results:
            new_results['silver score'] = old_results['silver score']
            new_results['silver params'] = old_results['silver params']
            new_results['silver time'] = old_results['silver time']
            new_results['silver method'] = old_results['silver method']
            new_results['silver run date'] = old_results['silver run date']
    elif 'best score' in old_results and ('silver params' not in old_results or old_results['best score'] > old_results['silver score']):
        new_results['silver score'] = old_results['best score']
        new_results['silver params'] = old_results['best params']
        new_results['silver time'] = old_results['best time']
        new_results['silver method'] = old_results['best method']
        new_results['silver run date'] = old_results['best run date']
    else:
        if 'silver score' in old_results:
            new_results['silver score'] = old_results['silver score']
            new_results['silver params'] = old_results['silver params']
            new_results['silver time'] = old_results['silver time']
            new_results['silver method'] = old_results['silver method']
            new_results['silver run date'] = old_results['silver run date']
        else:
            print('(debug:do nothing)')

    new_results['best score'] = new_results['_score']
    new_results['best time'] = new_results['_train time']
    new_results['best params'] = new_results['_params']
    new_results['best method'] = new_results['_method']
    new_results['best run date'] = new_results['date']
    new_results['suboptimal'] = 'pending'


def replace_new_in_best(new_results, old_results):

    new_results['best score'] = new_results['_score']
    new_results['best time'] = new_results['_train time']
    new_results['best params'] = new_results['_params']
    new_results['best method'] = new_results['_method']
    new_results['best run date'] = new_results['date']
    new_results['suboptimal'] = 'pending'


def assert_ok(updated, expected__json_filename, testname=""):
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
        assert failed == []


def test_module():
    if False:
        pass
    elif False:
        trial_df = pd.read_csv('data/final/df_listings_v09.csv')
        feature_engineer(trial_df, version=3)
    elif True:
        using_test_framework = True

        updated = {}
        update_results(updated, make_result(score=10, time=0.1, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected1.json', testname='test 1')

        update_results(updated, make_result(score=9, time=0.09, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected1B.json', testname='test 1B')
        #update_results(updated, make_result(score=11, time=0.11, method="method"), key='TEST', directory='./offline/')
        #if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected1C.json', testname='test 1C')

        update_results(updated, make_result(score=5, time=0.05, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected2.json', testname='test 2')

        update_results(updated, make_result(score=20, time=0.2, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected3.json', testname='test 3')

        update_results(updated, make_result(score=1, time=0.01, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected4.json', testname='test 4')

        update_results(updated, make_result(score=20, time=0.2, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected5.json', testname='test 5')

        update_results(updated, make_result(score=20, time=0.2, vary='_vary', method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected6.json', testname='test 6')

        update_results(updated, make_result(score=20, time=200.0, vary='_vary', method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected7.json', testname='test 7')

        update_results(updated, make_result(score=20, time=0.00002, vary='_vary', method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected8.json', testname='test 8')

        update_results(updated, make_result(score=20, time=0.02, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected9.json', testname='test 9')

        update_results(updated, make_result(score=20, time=0.00002, method="method"), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected10.json', testname='test 10')

    elif False:
        get_hyperparameters('catboost', False)
    else:
        pass


if __name__ == '__main__':
    test_module()
