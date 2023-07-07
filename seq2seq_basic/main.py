import torch
import torch.nn as nn
import random
random.seed(420)
from torch.utils.tensorboard import SummaryWriter
from dataset import german_vocab, english_vocab, train_iterator, valid_iterator, test_iterator, en_pad_idx, en_unk_idx
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, get_tf_ratio
from tqdm import tqdm


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
		# print("Input shape: ", x.shape)

		# embed -> batch_size, sequences, embed_size
		embed = self.dropout(self.embed_layer(x))
		# print("Embedding shape: ", embed.shape)

		# out - Hidden state at each timestep: (b_size, timesteps, hidden_size)
		# _ : Final hidden state and cell state (for each LSTM layer)
		out, (hidden, cell) = self.lstm(embed)
		return hidden, cell


class Decoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_stacked_layers, embed_dropout, rnn_dropout):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_stacked_layers = num_stacked_layers
		self.embed_layer = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_stacked_layers, batch_first=True, dropout=rnn_dropout)
		self.dropout = nn.Dropout(embed_dropout)
		self.fc = nn.Linear(hidden_size, vocab_size)

	def forward(self, x, hidden, cell):
		# x -> (batch_size, 1)
		# print("Input shape: ", x.shape)

		# embed -> (batch_size, 1, embed_size)
		embed = self.dropout(self.embed_layer(x))
		# print("Embedding shape: ", embed.shape)

		# hidden -> (num_stacked_layers, batch_size, hidden_state)
		out, (hidden, cell) = self.lstm(embed, (hidden, cell))
		# print("Output: ", out.shape)

		# out -> (batch_size, sequences, hidden_state)
		# x -> (batch_size, output_vocab_size)
		x = self.fc(out).squeeze()
		return x, hidden, cell


class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, source, target, teacher_force_ratio):
		# source -> (batch_size, sequences)
		b_size = source.shape[0]
		target_len = target.shape[1]
		target_vocab_size = len(english_vocab)

		outputs = torch.zeros(target_len, b_size, target_vocab_size).to(device)

		# Encode the source sentence
		hidden, cell = self.encoder(source)

		# First token
		x = target[:, 0]

		for t in range(1, target_len):
			output, hidden, cell = self.decoder(x.unsqueeze(1), hidden, cell)
			outputs[t] = output

			# predicted_tokens (indices) -> (batch_size,)
			predicted_tokens = output.argmax(1)
			x = target[:, t] if random.random() < teacher_force_ratio else predicted_tokens
		return outputs


if __name__ == "__main__":
	# Model hyperparameters
	load_model = False
	init_lr = 0.01
	epochs = 25
	device = "cpu"
	input_encoder_size = len(german_vocab)
	input_decoder_size = len(english_vocab)
	encoder_embedding_size = 300
	decoder_embedding_size = 300
	hidden_size = 1536
	lstm_stacked_layers = 1
	embed_dropout = 0.5
	lstm_dropout = 0.0

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
		lstm_stacked_layers,
		embed_dropout,
		lstm_dropout
	).to(device)

	# Wrapper model
	model = Seq2Seq(encoder_net, decoder_net).to(device)
	loss_function = nn.CrossEntropyLoss(ignore_index=en_pad_idx)
	optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

	if load_model:
		load_checkpoint(torch.load("my_checkpoint_10.pth.tar"), model, optimizer)

	# Tensorboard
	writer = SummaryWriter("runs/loss")
	step = 0

	for epoch in range(epochs):
		# Set to train mode
		model.train()

		# Get the teacher force ratio
		tf_ratio = get_tf_ratio(epoch)

		print(f"Epoch: {epoch}/{epochs}, tf_ratio: {tf_ratio}")

		for src, trg, _, _ in tqdm(train_iterator):
			inp_data = src.permute(1, 0).to(device)  # (batch_size, src_len)
			target = trg.permute(1, 0).to(device)  # (batch_size, trg_len)

			output = model(inp_data, target, tf_ratio)
			output = output[1:, :, :].reshape(-1, output.shape[2])  # (trg_len * batch_size, output_vocab_size)
			target = target[:, 1:].reshape(-1)  # (trg_len * batch_size)

			optimizer.zero_grad()
			loss = loss_function(output, target)
			loss.backward()

			# Clipping gradients
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

			optimizer.step()

			# Plot to tensorboard
			writer.add_scalar("Training loss", loss, global_step=step)
			step += 1

		model.eval()
		print(' '.join(translate_sentence(model, "Das ist ein wunderschönes Gemälde", german_vocab, english_vocab, "cpu")))

		# Save model
		checkpoint = {"state_dict": model.state_dict(), "optim": optimizer.state_dict()}
		save_checkpoint(checkpoint, f"my_checkpoint_{epoch}.pth.tar")


	






