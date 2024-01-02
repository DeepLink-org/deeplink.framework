from utils.test_in_subprocess import run_individual_test_cases

def will_fail():
    import torch
    print("hello")
    assert False

if __name__ == "__main__":
    run_individual_test_cases([
        will_fail,
        will_fail,
        will_fail,
    ], in_parallel=True)
