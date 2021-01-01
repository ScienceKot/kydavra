'''
Created with love by Sigmoid

@Author - Păpăluță Vasile - vpapaluta06@gmail.com
'''
class NotBetweenZeroAndOneError(BaseException):
    ''' Raised when a value is not between zero and one, but it should be'''
    pass

class NotBinaryData(BaseException):
    ''' Raised when the data passed is not binary '''
    pass

class NoSuchMethodError(BaseException):
    ''' Raised when the selector or reducer doesnt't have a method '''
    pass

