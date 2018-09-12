import numpy as np

class AttributesCelebA(object):
    def __init__(self, attrib_f):
        with open(attrib_f) as f:
            self.nfiles = int(f.readline().rstrip())
            #print(self.nfiles)
            attribline = f.readline().rstrip()
            self.attributes = attribline.split(' ')
            self.nAttributes = len(self.attributes)
            self.labels = np.zeros((self.nfiles, self.nAttributes), dtype=int)
            for i in range(1,self.nfiles+1):
                line = f.readline().rstrip()
                # replace multiple whitespace with one...
                line = ' '.join(line.split())
                tokens = line.split(' ')
                tag = tokens[0].split('.')
                idx = int(tag[0])-1
                thislabel = map(int, tokens[1:])
                self.labels[idx,:] = np.asarray(thislabel)
    def getLabels(self, labname):
        label_idx = self.attributes.index(labname)
        return self.labels[:,label_idx]
    def printSummary(self):
        print('num files = %s' % self.nfiles)
        for label in self.attributes:
            label_idx = self.attributes.index(label)
            s = np.sum(self.labels[:,label_idx])
            pos = (self.nfiles + s)/2
            neg = (self.nfiles - s)/2
            print('%s: %.2f' % (label, pos*100.0/(self.nfiles)))

if __name__ == '__main__':
    labels = AttributesCelebA('/home2/dataset/Celeb-A/list_attr_celeba.txt')
    labels.printSummary()
    smiling_labels = labels.getLabels('Smiling')
    print(smiling_labels)
    print(len(smiling_labels))
