import numpy as np
def create_dynamic_train_set():
	filename = 'database/bodmas/bodmas.npz'
	data = np.load(filename)
	X = data["X"]
	y = data["y"]

	# Indices by class
	idx_malware = np.where(y == 1)[0]
	idx_benign = np.where(y == 0)[0]

	# Random but reproducible
	np.random.seed(42)

	idx_malware_sel = np.random.choice(idx_malware, 50000, replace=False)
	idx_benign_sel = np.random.choice(idx_benign, 50000, replace=False)

	# Combine & shuffle
	selected_idx = np.concatenate([idx_malware_sel, idx_benign_sel])
	np.random.shuffle(selected_idx)

	X_balanced = X[selected_idx]
	y_balanced = y[selected_idx]

	print("Balanced X shape:", X_balanced.shape)
	print("Balanced y distribution:", {
		0: (y_balanced == 0).sum(),
		1: (y_balanced == 1).sum()
	})

	np.savez(
		"database/dynamic_training_data.npz",
		X=X_balanced,
		y=y_balanced
	)

if __name__ == "__main__":
	create_dynamic_train_set()