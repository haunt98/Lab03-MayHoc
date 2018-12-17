class Dataset:
    def __init__(self):
        # luu ten attribute cung thu tu xuat hien cua attribute
        self.attr = {}
        # luu danh sach data
        self.data = []
        # mac dinh khong debug
        self.debugState = False

    def debug(self, debugState):
        self.debugState = debugState

    def read_arff(self, filename):
        with open(filename, 'r') as f:
            # read @attribute
            attr_index = 0
            for line in f:
                if line.startswith("@attribute"):
                    # example
                    # @attribute hair {0,1}
                    self.attr[line.split()[1]] = attr_index
                    attr_index += 1
                elif line.startswith("@data"):
                    break
            # read @data
            for line in f:
                # example
                # aardvark,1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1,Mammals
                self.data.append(line.rstrip("\n").split(","))

            if self.debugState:
                print('attribute name')
                print(self.attr)

    # file test phai co attribute trung hop voi attribute cua file dataset
    def read_test(self, filename, attrNames, classifyAttrName):
        with open(filename, 'r') as f:
            test_data = []
            for line in f:
                if line.startswith("@data"):
                    break
            for line in f:
                test_data.append(line.rstrip("\n").split(","))

            # find out type of x
            for x in test_data:
                print(x)
                print(classifyAttrName, ": ", self.NaiveBayesClassify(
                    x, attrNames, classifyAttrName))

    # dem so lan attrValue xuat hien
    def countAttrValue(self, attrName, attrValue):
        count = 0
        for i in range(len(self.data)):
            if self.data[i][self.attr[attrName]] == attrValue:
                count += 1
        return count

    # xac suat attrValue xuat hien
    def probabilityAttrValue(self, attrName, attrValue):
        return self.countAttrValue(attrName, attrValue) / len(self.data)

    # so lan attrValue xuat hien neu biet knowAttrValue
    def countAttrValueWithKnowAttr(self, attrName, attrValue, knowAttrName, knowAttrValue):
        count = 0
        for i in range(len(self.data)):
            if self.data[i][self.attr[attrName]] == attrValue \
                    and self.data[i][self.attr[knowAttrName]] == knowAttrValue:
                count += 1
        return count

    # xac suat attrValue xuat hien neu biet knowAttrValue
    # Multinomial Naive Bayes
    # p(xi | c) = so lan xi xuat hien khi c xuat hien / so lan c xuat hien
    def probabilityAttrValueWithKnowAttr(self, attrName, attrValue, knowAttrName, knowAttrValue):
        # laplace smoothing
        return (self.countAttrValueWithKnowAttr(attrName, attrValue, knowAttrName, knowAttrValue) + 1) \
            / (self.countAttrValue(knowAttrName, knowAttrValue) + self.countDisctintValueInAttr(attrName))

    # dem tap gia tri cua attr
    # 'hair' la 2
    def countDisctintValueInAttr(self, attrName):
        return len(self.getDisctintValueInAttr(attrName))

    # tra ve list tap gia tri cua attr
    # example
    # 'hair' la ['0', '1']
    def getDisctintValueInAttr(self, attrName):
        value = []
        for i in range(len(self.data)):
            value.append(self.data[i][self.attr[attrName]])
        return set(value)

    # x la data duoc test
    # cac thuoc tinh x co nam trong attrNames
    # classifyAttrName la thuoc tinh can xac dinh trong x
    # p(c|x) = p(x|c)p(c) = p(xi|c)...p(c)
    def NaiveBayesClassify(self, x, attrNames, classifyAttrName):
        classifyAttrValues = self.getDisctintValueInAttr(classifyAttrName)
        classifyProbality = {}
        for classifyAttrValue in classifyAttrValues:
            classifyProbality[classifyAttrValue] = 1
            for attrName in attrNames:
                # p(xi)
                classifyProbality[classifyAttrValue] *= \
                    self.probabilityAttrValueWithKnowAttr(
                        attrName, x[self.attr[attrName]], classifyAttrName, classifyAttrValue)
            # p(c)
            classifyProbality[classifyAttrValue] *= self.probabilityAttrValue(
                classifyAttrName, classifyAttrValue)
        renameClassifyAttrName = max(
            classifyProbality, key=classifyProbality.get)

        if self.debugState:
            print(sorted(classifyProbality.items()))

        return renameClassifyAttrName


def main():
    zoo = Dataset()
    zoo.debug(False)
    zoo.read_arff("zoo.arff")
    zoo.read_test("zootest.arff", list(zoo.attr.keys())[1:-1], 'type')


if __name__ == '__main__':
    main()
