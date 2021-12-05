
def save(path, name, lis):
    file = open(path + name + '.csv', 'w+')
    for i in range(len(lis)):
        file.write(str(lis[i]) + "\n")
    file.close()