class ProgressBar:
    def __init__(self, empty_char='-', fill_char='=', max_value=100, min_value=0, value=0, length=50):
        '''
        Args:
            empty_char : String, empty space in bar
            fill_char : String, filled space in bar
            max_value : Integer, maximum value of bar
            min_value : Integer, minimum value of bar
            value : Integer, inital value of bar
            length : Integer, visual length of bar
        '''
        self.empty_char = empty_char
        self.fill_char = fill_char
        self.max_value = max_value
        self.min_value = min_value
        self.value = value
        self.length = length
    
    def update(self, increment):
        '''
        Updates and returns the progress bar

        Args:
            increment : int, how much the internal value will be changed by
        
        Returns:
            String : visual progress bar (make sure to use end='\r' in print statements)
        '''
        self.value += increment

        if self.value > self.max_value:
            self.value = self.max_value
        if self.value < self.min_value:
            self.value = self.min_value

        return self
    
    def __str__(self):
        block_num = int(self.value//(self.max_value/self.length))
        return f'[{self.fill_char*block_num + self.empty_char*(self.length-block_num)}]'