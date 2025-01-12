# datasets_with_img
这段代码实现了一个增强版的 `FakeNewsDataset` 类，它不仅处理文本数据，还同时处理图像数据。它结合了文本处理（分词和编码）和图像处理（图像读取和转换），并为批量处理提供了一个 `create_mini_batch` 函数。以下是详细的解释：

### `FakeNewsDataset` 类

#### 1. **初始化方法 `__init__`**:
   - `mode`: 数据集的模式，可以是 `'train'` 或 `'test'`。
   - `datasets`: 数据集的名称（例如 `'fake_news'`）。
   - `tokenize`: 用于处理文本的分词器（通常是 BERT 或其他模型的 tokenizer）。
   - `path`: CSV 文件的路径，用于加载数据集的文本内容。
   - `img_path`: 图像文件的路径，用于加载图像。

   - `assert mode in ['train', 'test']`: 确保 `mode` 参数值合法。
   - `self.img_path = img_path`: 保存图像文件的路径。
   - `self.df = pd.read_csv(path + datasets + '_' + mode + '.csv').fillna('')`: 加载包含假新闻数据集的 CSV 文件，并填充所有缺失的值。
   - `self.len = len(self.df)`: 记录数据集的样本数量。
   - `self.tokenizer = tokenize`: 存储传入的分词器对象。

#### 2. **`__getitem__` 方法**:
   - 输入：索引 `idx`，返回该位置的数据。
   - `statement = self.df.iloc[idx]['content']`: 从数据框中获取文本内容（假新闻的正文）。
   - `label = self.df.iloc[idx]['label']`: 获取该样本的标签（假新闻标签）。
   - `img = self.df.iloc[idx]['image']`: 获取图像文件名。

   - **文本处理**:
     - `word_pieces = ['[CLS]']`: 在文本前添加 `[CLS]` 标记，这是许多预训练模型（如 BERT）要求的。
     - `statement = self.tokenizer.tokenize(statement)`: 使用 tokenizer 对文本进行分词。
     - 如果分词后的文本长度超过 100，截断文本至前 100 个词。
     - `word_pieces += statement + ['[SEP]']`: 在文本末尾添加 `[SEP]` 标记，标记句子的结束。
     - `ids = self.tokenizer.convert_tokens_to_ids(word_pieces)`: 将分词后的单词转换为 token ID。
     - `tokens_tensor = torch.tensor(ids)`: 将 token ID 转换为张量。
     - `segments_tensor = torch.tensor([0] * len_st, dtype=torch.long)`: 创建与文本长度相同的 segment ID，通常用来表示不同句子的关系（此处设为 0）。

   - **图像处理**:
     - `image = Image.open(self.img_path + img)`: 使用 PIL 加载图像文件。
     - `image_tensor = transforms.ToTensor()(image)`: 使用 `ToTensor` 转换图像为 PyTorch 张量。

   - 返回：(token_tensor, segments_tensor, image_tensor, label_tensor)，即返回文本数据的 token 张量、segment 张量、图像张量和标签张量。

#### 3. **`__len__` 方法**:
   - 返回数据集的长度（即样本的数量）。

### `create_mini_batch` 函数

该函数用于将多个样本整合为一个批次，并对数据进行处理以便输入到模型中。

#### 1. **输入**:
   - `samples`: 这是一个样本列表，每个样本是 `(tokens_tensor, segments_tensor, image_tensor, label_tensor)`。

#### 2. **处理步骤**:
   - `tokens_tensors = [s[0] for s in samples]`: 提取所有样本的 token 张量。
   - `segments_tensors = [s[1] for s in samples]`: 提取所有样本的 segment 张量。
   - `imgs_tensors = [s[2] for s in samples]`: 提取所有样本的图像张量。
   - `imgs_tensors = torch.stack(imgs_tensors)`: 使用 `torch.stack` 将图像张量堆叠成一个批次。

   - **标签处理**:
     - `label_ids = torch.stack([s[3] for s in samples])`: 如果样本中包含标签，将标签堆叠成一个张量。
     - 如果标签是 `None`，则设置 `label_ids = None`。

   - **填充处理**:
     - `tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)`: 使用 `pad_sequence` 对 token 张量进行填充，使得批次中所有文本的长度相同。
     - `segments_tensors = pad_sequence(segments_tensors, batch_first=True)`: 对 segment 张量进行填充。
   
   - **掩码处理**:
     - `masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)`: 创建一个与 `tokens_tensors` 相同形状的全零张量，用于表示掩码。
     - `masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)`: 将所有非零位置（有效 token）填充为 1，表示有效的部分。

#### 3. **返回**:
   - 返回处理后的批次数据：`tokens_tensors`, `segments_tensors`, `masks_tensors`, `imgs_tensors`, `label_ids`。

### 总结

- `FakeNewsDataset` 类是一个自定义数据集类，用于加载和处理包含文本和图像的假新闻数据。它支持文本分词、图像处理（使用 PIL 和 `transforms`）以及将这些数据转换为张量。
- `create_mini_batch` 函数将多个样本合并成一个批次，处理了文本数据的填充、掩码和图像张量的堆叠。
  
这样，您可以在训练中同时输入文本和图像数据，结合文本和视觉信息进行假新闻检测或其他任务。

# qnn
这段代码实现了一个量子卷积神经网络（QCNN），结合了量子电路和经典神经网络层，适用于量子机器学习任务。接下来，我们逐步解析每个部分的代码。

### 关键组件：

1. **量子电路准备与层**：
    - **`state_prepare(input, qbits)`**:
        - 该函数使用 **AngleEmbedding** 对输入数据进行编码，准备量子态。
        - 输入 `input` 是一个数据张量，`qbits` 表示量子比特的数量，函数会根据这两个参数生成一个量子态。
        - `AngleEmbedding` 是将经典数据映射到量子态的过程，通常通过在量子比特上执行旋转门来实现。

    - **`conv(b1, b2, params)`**:
        - 该函数定义了一个 **量子卷积层**，该层包含多个量子门操作：
            - `RZ`：绕 Z 轴的旋转门，根据给定的角度旋转量子比特。
            - `CNOT`：控制非门，通常用于量子比特之间的纠缠。
            - `RY`：绕 Y 轴的旋转门，根据给定的角度旋转量子比特。
        - 该函数接收两个量子比特 `b1` 和 `b2`，以及一组参数 `params`。通过执行一系列的旋转和纠缠操作，卷积层将输入数据转换为量子态。

    - **`pool(b1, b2, params)`**:
        - 这个函数定义了一个 **量子池化层**，其操作和卷积层相似，但主要用于在量子态上进行某种类型的“下采样”或聚合。
        - 与 `conv` 类似，`pool` 也使用 `RZ`、`RY` 和 `CNOT` 门来处理量子比特。

2. **量子电路 `qcnn4` 和 `qcnn8`**：
    - **`qcnn4`**：这是一个包含 4 个量子比特的量子电路，适用于具有 4 个量子比特的输入数据。电路由多个卷积层和池化层组成。
    - **`qcnn8`**：这是一个包含 8 个量子比特的量子电路，适用于具有 8 个量子比特的输入数据。它的结构比 `qcnn4` 更复杂，包含更多的卷积和池化层。

    在这两个量子电路中，`qml.qnode` 装饰器用于将它们转化为量子节点，允许与 PyTorch 张量兼容，并支持反向传播 (`backprop`) 用于梯度计算。

3. **量子电路绘图**：
    - **`draw(input, params, cir, name)`**：
        - 该函数用于绘制量子电路，并将结果保存为图像文件。它使用 PennyLane 的绘图工具 `qml.drawer.use_style('pennylane')` 和 `qml.draw_mpl()` 来可视化量子电路结构，并将图像保存为 PNG 格式。

4. **`QCNN` 类**：
    - 这是一个 PyTorch 神经网络模型类，继承自 `nn.Module`。
    - **`__init__(self, qbits)`**：初始化时，`qbits` 决定量子电路的量子比特数，如果 `qbits` 为 4，则选择 `qcnn4` 电路，如果为 8，则选择 `qcnn8` 电路。
    - **`forward(self, x)`**：
        - `forward` 方法定义了数据如何通过网络传递。在这个方法中：
            - 输入 `x` 首先进行了归一化处理：通过 `torch.max` 和 `torch.min` 将数据规范化到 `[0, 1]` 区间，然后将其缩放到 `[0, 2π]` 区间，这是量子电路常见的输入格式。
            - 然后，将处理后的输入传递给量子电路 (`self.qcnn_layer(x)`)，并返回量子电路的输出。

### 总结：

- 该代码实现了一个量子卷积神经网络（QCNN），它结合了量子计算和深度学习，通过量子电路处理数据，并用量子卷积层和池化层提取特征。
- 量子电路部分使用 PennyLane 库定义，包括量子门（如 `RZ`、`RY`、`CNOT`）和量子层（如卷积和池化）。
- 网络接受输入数据（如图像或其他数据），通过量子电路进行处理，并输出概率分布。
- 通过 `QCNN` 类将量子电路集成到 PyTorch 神经网络模型中，支持使用反向传播进行训练。

# multimodal_train
这段代码定义了一个集成了 XLNet、VGG 和量子卷积神经网络（QCNN）的混合模型，结合了自然语言处理（NLP）和计算机视觉（CV）来进行假新闻检测任务。具体来说，模型同时处理文本数据（通过 XLNet）和图像数据（通过 VGG），然后将其输出与量子计算的部分（QCNN）结合进行进一步处理。以下是详细的代码解析：

### 1. **数据准备**

- `FakeNewsDataset` 类和 `create_mini_batch` 函数被用来加载和处理数据集（包含文本和图像）。这里假设数据集包含文本和图像，其中：
  - 文本通过 XLNet 进行处理。
  - 图像通过 VGG 网络进行处理。
  
- `BATCH_SIZE` 和 `NUM_EPOCHS` 设置了训练过程中的批次大小和训练轮数。

### 2. **模型选择和加载**

#### 2.1 **XLNet（文本处理）**
- `AutoTokenizer` 和 `XLNetForSequenceClassification` 通过 Hugging Face Transformers 加载了一个预训练的 XLNet 模型。
  - `tokenizer` 用于对文本进行分词。
  - `xlnet` 是预训练的 XLNet 模型，能够处理文本数据，并进行分类。

#### 2.2 **VGG（图像处理）**
- `vgg` 是从 torchvision 加载的预训练 VGG-19 模型。最后一层 `classifier[6]` 被替换为一个线性层，将输出大小设置为 `qbits//2`，即根据量子比特的数量（`qbits=8`，所以输出为 4）。

#### 2.3 **量子卷积神经网络（QCNN）**
- `qcnn` 是一个量子卷积神经网络，接受来自 XLNet 和 VGG 的融合输出，进行后续处理。

### 3. **训练过程设置**

#### 3.1 **设备配置**
- 根据是否存在 GPU，设置 `device` 为 `'cuda:0'` 或 `'cpu'`，并将模型移动到相应的设备上（GPU 或 CPU）。

#### 3.2 **优化器和损失函数**
- 优化器使用 `AdamW`，它同时优化三个部分的参数：
  - XLNet 的参数
  - VGG 的参数
  - QCNN 的参数
- 损失函数使用 `CrossEntropyLoss`，适合多分类任务。

#### 3.3 **训练过程**

- **数据加载**：
  - 使用 `FakeNewsDataset` 加载训练数据，并通过 `DataLoader` 进行批处理。
  - 每个批次的数据包括文本数据（token、segment、mask）、图像数据和标签。

- **前向传播**：
  - 对文本数据：XLNet 模型输出文本的嵌入。
  - 对图像数据：VGG 模型输出图像的特征。
  - 将这两部分特征拼接（`torch.cat`）在一起，作为输入传递给量子卷积神经网络（QCNN）。

- **损失计算和反向传播**：
  - 计算模型的输出和真实标签之间的损失。
  - 使用反向传播（`loss.backward()`）更新模型的参数。

- **准确度计算**：
  - 使用 `accuracy_score` 计算当前批次的准确度，并记录在 `train_acc` 中。

#### 3.4 **训练过程输出和保存**
- 每个 epoch 会显示当前的训练损失和准确度，并保存每个 epoch 的训练损失和准确度到文件。
  - `loss_v` 和 `acc_v` 分别保存训练过程中的损失值和准确度。
  - 最终的损失和准确度会被保存为文件 `loss_v.pth` 和 `acc_v.pth`。

### 4. **总体流程**

- **输入数据**：每个训练样本包含文本和图像数据。
  - 文本数据由 XLNet 处理，输出文本嵌入。
  - 图像数据由 VGG 处理，输出图像特征。
- **数据融合**：将文本和图像的特征拼接在一起，作为量子卷积神经网络的输入。
- **量子卷积神经网络**：QCNN 接收这些融合后的特征，并进行后续处理，生成分类输出。
- **训练**：通过 `AdamW` 优化器对模型进行训练，目标是最小化分类损失。

### 5. **注意事项和改进点**

- **量子部分**：代码中使用了 `qnn.QCNN` 来进行量子卷积，但具体实现细节没有给出。如果 `qnn` 是自定义的量子神经网络类，确保量子部分的梯度能够正确计算。
- **数据加载**：在处理图像数据时，确保 `FakeNewsDataset` 和 `create_mini_batch` 函数能够正确加载和预处理数据。特别是图像的尺寸和格式需要与 VGG 的输入要求相匹配。
- **混合模型的训练**：文本和图像的特征融合部分需要合理设计，确保两种不同模态的数据能够有效结合并提供有用的特征。

### 总结
这段代码实现了一个结合了量子计算、自然语言处理（NLP）和计算机视觉（CV）的混合模型，用于假新闻分类任务。它通过 XLNet 处理文本数据，通过 VGG 处理图像数据，并利用量子卷积神经网络（QCNN）进行最终的特征提取和分类。

# textonly_train
这段代码实现了一个多模态模型训练过程，结合了 **XLNet**（文本处理）、**量子卷积神经网络（QCNN）**（量子部分）来进行假新闻检测任务。以下是对代码的详细解析：

### 1. **模型和数据准备**

#### 1.1 **XLNet 和 Tokenizer**
- 使用 `AutoTokenizer` 加载预训练的 `xlnet-base-cased` 分词器。该分词器将文本输入转换为适合输入到模型的格式。
- 使用 `XLNetForSequenceClassification` 加载预训练的 XLNet 模型，它被用于文本分类任务。这里的模型输出用于后续与量子模型（QCNN）进行结合。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, proxies=proxy)

cla_model = XLNetForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4, proxies=proxy
)
```
- `MODEL_NAME` 是使用的预训练模型的名称，这里使用的是 `"xlnet-base-cased"`，适用于文本分类任务。
- `num_labels=4` 表示这是一个四分类任务（例如：假新闻检测中的四个类别）。

#### 1.2 **量子卷积神经网络（QCNN）**
- `qcnn = qnn.QCNN(4)`：实例化一个量子卷积神经网络（QCNN）模型，`4` 表示量子比特的数量。

#### 1.3 **数据加载**
- 数据集加载通过 `FakeNewsDataset` 类和 `DataLoader` 完成。这里加载的是 `gossip` 数据集中的训练数据。
- `BATCH_SIZE` 设置为 56，每次训练使用 56 个样本。`create_mini_batch` 是用于处理每个批次的函数，它确保数据符合模型的输入要求（文本和图像）。

```python
trainset = FakeNewsDataset(
    "train", tokenizer=tokenizer, path=data_path, datasets="gossip"
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
```

### 2. **训练过程**

#### 2.1 **设备设置**
- `device` 是检查是否有 GPU 可用，并将模型移动到适当的设备（CUDA 或 CPU）。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
- `cla_model`（XLNet）和 `qcnn`（量子卷积神经网络）都被移动到适当的设备上。

#### 2.2 **优化器与损失函数**
- 使用 `AdamW` 优化器，分别优化 `cla_model` 和 `qcnn` 的参数。
- 使用 `CrossEntropyLoss` 作为损失函数，这对于多分类问题是标准的损失函数。

```python
optimizer = torch.optim.AdamW(
    [{"params": cla_model.parameters()}, {"params": qcnn.parameters()}], lr=1e-3
)

loss_func = nn.CrossEntropyLoss()
```

#### 2.3 **训练循环**
- 在每个 epoch 中，训练数据被加载并传递给模型。
- 对于每个批次：
  - **文本处理**：使用 `cla_model`（XLNet）处理输入的文本数据，得到文本特征表示。
  - **量子卷积神经网络（QCNN）**：将 XLNet 的输出与量子卷积网络（QCNN）的输入拼接（`q_input`），并传递给 QCNN 进行进一步处理。
  - **损失计算和反向传播**：计算模型输出与真实标签之间的损失，并通过反向传播更新模型的参数。
  - **准确度计算**：使用 `accuracy_score` 计算当前批次的分类准确率。

```python
for epoch in range(NUM_EPOCHS):
    train_loss = 0.0
    train_acc = 0.0

    loop = tqdm(trainloader)
    for batch_idx, data in enumerate(loop):
        tokens_tensors, segments_tensors, masks_tensors, labels = [
            t.to(device) for t in data
        ]

        optimizer.zero_grad()
        q_input = cla_model(
            input_ids=tokens_tensors,
            token_type_ids=segments_tensors,
            attention_mask=masks_tensors,
        )[0]

        outputs = qcnn(q_input)

        loss = loss_func(outputs, labels)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1)
        train_acc = accuracy_score(pred.cpu().tolist(), labels.cpu().tolist())

        train_loss += loss.item()

        loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        loop.set_postfix(acc=train_acc, loss=train_loss / (batch_idx + 1))
```

- **进度条**：使用 `tqdm` 显示训练的进度，包括当前的 epoch、损失和准确率。

#### 2.4 **模型保存**
- 每个 epoch 结束时，保存 `cla_model`（XLNet）和 `qcnn` 的模型权重到文件中。

```python
torch.save(cla_model, "result/xlnet_politi_tx.pth")
torch.save(qcnn.state_dict(), "result/qcnn_politi_tx.pth")
```

- `torch.save` 将训练后的模型权重保存到文件，以便后续使用或评估。

### 3. **总结**

- **数据**：使用了包含文本和图像数据的假新闻数据集。
- **模型**：
  - `XLNet` 用于处理文本数据，进行文本特征提取。
  - `qcnn`（量子卷积神经网络）接收 `XLNet` 输出的文本特征，并进行进一步处理。
- **训练**：使用标准的训练流程，通过 `AdamW` 优化器和 `CrossEntropyLoss` 损失函数进行训练。
- **保存模型**：训练过程中定期保存模型，便于后续加载和推理。

### 4. **注意事项**
- **代理设置**：代理 `proxy = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}` 用于在需要通过代理服务器访问 Hugging Face Hub 或其他外部资源时使用。
- **量子计算部分**：这段代码的量子卷积部分（`qcnn`）假设已经正确实现并能够处理输入数据。请确保量子模型的训练过程与经典模型兼容，且能够计算梯度以进行反向传播。

整体而言，这段代码展示了一个混合经典-量子神经网络模型的训练流程，结合了深度学习和量子计算，适用于处理包括文本和图像数据的任务。
