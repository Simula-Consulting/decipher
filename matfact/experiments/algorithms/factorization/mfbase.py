from abc import ABC, abstractmethod


class BaseMF(ABC):
	"Base class for matrix factorization algorithms."

	@property 
	@abstractmethod 
	def M(self):
		return 

	@abstractmethod 
	def run_step(self):
		return 

	@abstractmethod 
	def loss(self):
		return 

	def test(self):
		print(f"I'm {self.__class__.__name__}, and my M is {self.M}")
