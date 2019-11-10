from rnn import main as rnn_main
from ffnn1fix import main as ffnn_main


FLAG = 'RNN'


def main():
	if FLAG == 'RNN':
		#raise NotImplementedError
		embedding_dim = 128
		hidden_dim = 64
		number_of_epochs = 10
		rnn_main(embedding_dim=embedding_dim, hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)
	elif FLAG == 'FFNN':
		hidden_dim = 32
		number_of_epochs = 10
		ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)



if __name__ == '__main__':
	main()
