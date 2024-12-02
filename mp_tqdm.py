from multiprocessing import Pool, Manager
from tqdm import tqdm

def my_func(x, progress):
    # some long-running operation
    result = x * 2
    progress.update(1)
    return result

if __name__ == '__main__':
    pool = Pool(processes=4)
    manager = Manager()
    progress = manager.tqdm(total=100)
    results = []
    for result in pool.imap_unordered(my_func, range(100)):
        results.append(result)
    progress.close()
