

class A:
    def __init__(self):
        self.a = 10



class B(A):
    def __init__(self, b):
        super(B, self).__init__()
        self.b = b
        print(self.a)



if __name__ == '__main__':
    B = B(2)
