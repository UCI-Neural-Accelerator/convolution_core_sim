file1 = open('output_convolver.txt', 'rt')
file2 = open('output_convolver_py.txt', 'rt')

i = 0
correct = 0
incorrect = 0

unit = file1.readlines()
python = file2.readlines()

while i < 576:
    if (unit[i]==python[i]):
        correct += 1
        #print(i)
    else:
        incorrect += 1

    i += 1

if (correct==576):
    print("CORRECT")
else:
    print("INCORRECT")

# print(correct)
# print(incorrect)

file1.close()
file2.close()