class sw(object):

    def __init__(self):

        self.__Positions = []
        self.__Gbest = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def _points(self, agents):
        self.__Positions.append([list(i) for i in agents])

    def get_agents(self):
        """Returns a history of all agents of the algorithm (return type:
        list)"""

        return self.__Positions

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""
        return list(self.__Gbest)
    
    def _set_zhistory(self, z):
        self.__zhistory = z
        
    def _set_time(self, t):
        self.__time = t
        
    def _set_history(self, Gb):
        self.__history = Gb
        
    def get_zhistory(self):
        """Return the best position of algorithm (return type: list)"""
        return list(self.__zhistory)
    def get_time(self):
        """Return the best position of algorithm (return type: list)"""
        return self.__time    
    def get_history(self):
        """Return the best position of algorithm (return type: list)"""
        return list(self.__history)
