from nltk.tree import Tree
import random

def create_example(width, numbers):
    parse = []
    for _ in range(width):
        num = random.choice(numbers)
        parse.append(Tree(num, [str(num)]))
    while len(parse) != 1:
        ind1 = random.randint(0,len(parse)-1)
        ind2 = random.randint(0,len(parse)-1)
        while ind2 == ind1:
            ind2 = random.randint(0,len(parse)-1)
        leaf1 = parse[ind1]
        leaf2 = parse[ind2]
        if ind2 > ind1:
            del parse[ind2]
            del parse[ind1]
        else:
            del parse[ind1]
            del parse[ind2]
        op = random.choice(["+", "-"])
        if op == "+":
            parse.append(Tree(leaf1.label() + leaf2.label(),[Tree(None,["("]), leaf1, Tree(None,["+"]), leaf2, Tree(None,[")"])]))
        if op == "-":
            parse.append(Tree(leaf1.label() - leaf2.label(),[Tree(None,["("]), leaf1, Tree(None,["-"]), leaf2, Tree(None,[")"])]))
    parse = parse[0]
    example = {"tree":parse, "input":" ".join(parse.leaves()), "output": parse.label()}
    return example


for _ in range(20):
    print(create_example(5, [0,1,2,3,4,5,6,7,8,9]))
