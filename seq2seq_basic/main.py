import torch
import torch.nn as nn
import random
random.seed(420)
from torch.utils.tensorboard import SummaryWriter
from dataset import german_vocab, english_vocab, train_iterator, valid_iterator, test_iterator, en_pad_idx
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


class Encoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_stacked_layers, embed_dropout, rnn_dropout):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_stacked_layers = num_stacked_layers
		self.embed_layer = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_stacked_layers, batch_first=True, dropout=rnn_dropout)
		self.dropout = nn.Dropout(embed_dropout)

	def forward(self, x):
		# x -> batch_size, sequences
		# embed -> batch_size, sequences, embed_size
		embed = self.dropout(self.embed_layer(x))

		b_size = x.shape[0]
		h0 = torch.zeros([self.num_stacked_layers, b_size, self.hidden_size])
		c0 = torch.zeros([self.num_stacked_layers, b_size, self.hidden_size])
		# out - Hidden state at each timestep: (b_size, timesteps, hidden_size)
        # _ : Final hidden state and cell state (for each LSTM layer)
		out, (hidden, cell) = self.lstm(embed, (h0, c0))
		return hidden, cell

class Decoder(nn.Module):
	def __init__(self, inp_vocab_size, out_vocab_size, input_size, hidden_size, num_stacked_layers, embed_dropout, rnn_dropout):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_stacked_layers = num_stacked_layers
		self.embed_layer = nn.Embedding(inp_vocab_size, input_size)
		self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, dropout=rnn_dropout)
		self.dropout = nn.Dropout(embed_dropout)
		self.fc = nn.Linear(hidden_size, out_vocab_size)

	def forward(self, x, hidden, cell):
		# x -> (batch_size, 1)
		# embed -> (batch_size, 1, embed_size)
		embed = self.dropout(self.embed_layer(x))
		print("Decoder Embedding Layer Shape: ", embed.shape)

		# hidden -> (num_stacked_layers, batch_size, hidden_state)
		out, (hidden, cell) = self.lstm(embed, (hidden, cell))

		# out -> (batch_size, sequences, hidden_state) 
		# x -> (batch_size, output_vocab_size)
		x = self.fc(out).squeeze()
		return x, hidden, cell

class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		
	def forward(self, source, target, teacher_force_ratio=0.5):
		# source -> (batch_size, sequences)
		b_size = source.shape[0]
		target_len = target.shape[1]
		target_vocab_size = len(english_vocab)

		outputs = torch.zeros(target_len, b_size, target_vocab_size).to(device)

		# Encode the source sentence
		hidden, cell = self.encoder(source)

		# First token
		x = target[0]

		for t in range(1, target_len):
			output, hidden, cell = self.decoder(x, hidden, cell)
			outputs[t] = output

			# predicted_tokens (indices) -> (batch_size,)
			predicted_tokens = output.argmax(1)
			x = target[t] if random.random() < teacher_force_ratio else predicted_tokens
		return outputs



if __name__ == "__main__":
	# model_enc = Encoder(160000, 128, 1000, 4, 0.2, 0.2)
	# # 2, 5, 1
	# inp = torch.tensor([[1, 10, 100, 500, 990], [1, 10, 100, 500, 990]])
	# hidden, cell = model_enc.forward(inp)
	# print("Hidden, Cell Encoder: ", hidden.shape, cell.shape)

	# model_dec = Decoder(160000, 80000, 128, 1000, 4, 0.2, 0.2)
	# inp = torch.tensor([[0], [0]])
	# probs, hidden, cell = model_dec.forward(inp, hidden, cell)
	# print("Probabilities, Hidden, Cell Decoder: ", probs.shape, hidden.shape, cell.shape)

	# model = Seq2Seq(encoder=Encoder(160000, 256, 1024, 4, 0.2, 0.2).to(device),
	# 	    decoder=Decoder(160000, 80000, 256, 1024, 4, 0.2, 0.2).to(device))
	# output = model.forward(
	# 	source = torch.randint(low=0, high=160000, size=(10, 20)),
	# 	target = torch.randint(low=0, high=80000, size=(25, 10)),
	# 	teacher_force_ratio = 0.5
	# )
	# print(output.shape)

	# Training hyperparameters
	epochs = 10
	bs = 4
	lr = 0.001

	# Model hyperparameters
	load_model = False
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	input_encoder_size = len(german_vocab)
	input_decoder_size = len(english_vocab)
	decoder_output_size = len(english_vocab)
	encoder_embedding_size = 300
	decoder_embedding_size = 300
	hidden_size = 1024
	lstm_stacked_layers = 2
	embed_dropout = 0.5
	lstm_dropout = 0.2
	
	# Tensorboard
	writer = SummaryWriter("runs/plot_loss")
	step = 0

	# Encoder
	encoder_net = Encoder(
	    input_encoder_size, 
	    encoder_embedding_size, 
	    hidden_size, 
	    lstm_stacked_layers, 
	    embed_dropout, 
	    lstm_dropout
	).to(device)

	# Decoder
	decoder_net = Decoder(
	    input_decoder_size,
	    decoder_embedding_size,
	    hidden_size,
	    decoder_output_size,
	    lstm_stacked_layers,
	    embed_dropout,
	    lstm_dropout
	).to(device)

	# Wrapper model
	model = Seq2Seq(encoder_net, decoder_net).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	loss = nn.CrossEntropyLoss(ignore_index=en_pad_idx)

	for epoch in range(epochs):
		print(f"Epoch: {epoch+1}/{epochs}")

		# Save model
		checkpoint = {"state_dict": model.state_dict(), "optim": optimizer.state_dict()}
		save_checkpoint(checkpoint, f"my_checkpoint_{epoch+1}.pth.tar")
		print("Checkpoint saved")

		for batch_idx, (src, trg, _, _) in enumerate(train_iterator):
			inp_data = src.permute(1, 0).to(device) # (batch_size, src_len)
    		target = trg.permute(1, 0).to(device) # (batch_size, trg_len)

			output = model(inp_data, target)  # (trg_len, batch_size, output_vocab_size)
			print(output).shape

			quit()


	






