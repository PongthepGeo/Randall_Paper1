data = pd.read_csv('datasets/well_logs.csv')

test_well = 'CROSS H CATTLE'
train = data.loc[data['Well Name'] != test_well]
test = data.loc[data['Well Name'] == test_well]
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
X = train.drop(drop_cols, axis=1) 
y = train['Facies'] 
X_test = test.drop(drop_cols, axis=1) 
y_test = test['Facies']

n_train_examples = int(len(train_data) * val_ratio)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_iterator = data.DataLoader(valid_data, batch_size=batch_size)
test_iterator = data.DataLoader(test_data, batch_size=batch_size)

###################################################################################

data = pd.read_csv('datasets/well_logs.csv')
data.fillna(-999, inplace=True)
le = LabelEncoder()
data['Facies'] = le.fit_transform(data['Facies'])

test_well = 'CROSS H CATTLE'
train = data.loc[data['Well Name'] != test_well]
test = data.loc[data['Well Name'] == test_well]
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
X_train = train.drop(drop_cols, axis=1) 
y_train = train['Facies'] 
X_test = test.drop(drop_cols, axis=1) 
y_test = test['Facies']

from sklearn.model_selection import StratifiedKFold

# Set the number of folds
num_folds = 5

# Initialize a StratifiedKFold object
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Split the data into folds and iterate over each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    
    # Get the training and validation data for this fold
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Convert data to tensors
    X_train_tensor = torch.Tensor(X_train_fold.values)
    y_train_tensor = torch.Tensor(y_train_fold.values).long()
    X_val_tensor = torch.Tensor(X_val_fold.values)
    y_val_tensor = torch.Tensor(y_val_fold.values).long()

    # Create PyTorch datasets
    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = data.TensorDataset(X_val_tensor, y_val_tensor)

    # Create PyTorch data loaders
    batch_size = 32
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    # Train the model for this fold
    model = DNN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        # Print the average training and validation loss for this epoch
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")
