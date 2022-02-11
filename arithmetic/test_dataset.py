from dataset import ArithmeticDataset2

def isin(vector, matrix):
    # NOTE: this is stupid, I wish there was a more sensible API for this
    return (vector == matrix).all(-1).any()

def check_overlap(train, test):
    count = 0
    for example in test.x:
        count += isin(example, train.x)

    return (count/len(test)*100)

if __name__ == "__main__":
    highest_number = 33
    train_size = 10000
    test_size = 1000

    d1 = ArithmeticDataset2(highest_number = highest_number, size=test_size, seed=42+1)
    d2 = ArithmeticDataset2(highest_number = highest_number, size=train_size, seed=42)

    count = 0
    for example in d1.x:
        count += isin(example, d2.x)

    print(count/test_size*100)
