from rnn import main as rnn_main
from ffnn import main as ffnn_main


FLAG = 'FFNN'


def main():
	if FLAG == 'RNN':
		raise NotImplementedError
		rnn_main()
	elif FLAG == 'FFNN':
		hidden_dim = 32
		number_of_epochs = 10
		ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)



if __name__ == '__main__':
	main()
