import torch
import torch.nn.functional as F
import time


def train(model, train_iter, dev_iter, test_iter, num_epochs, device='cpu', threshold=0.5, k=1):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.4e-2)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (inputs, labels) in enumerate(train_iter):
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.to(device)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels, reduction='mean')
            loss.backward()
            optimizer.step()

            if total_batch % 10 == 0:
                train_p_at_k = batch_p_at_k_sum(outputs, labels, threshold=threshold, k=k) / len(outputs)
                dev_p_at_k, dev_loss = evaluate(model, dev_iter, device=device, threshold=threshold, k=k)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    # last_improve = total_batch
                else:
                    improve = ''
                time_diff = get_time_diff(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train P@{2}: {3:>6.2%},  Dev Loss: {4:>5.2},  Dev P@{5}: ' \
                      '{6:>6.2%},  Time: {7} {8} '
                print(msg.format(total_batch, loss.item(), k, train_p_at_k, dev_loss, k, dev_p_at_k, time_diff, improve))
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train P@{2}: {3:>6.2%},  Time: {4}'
                # print(msg.format(total_batch, loss.item(), k, train_p_at_k, time_diff))
            total_batch += 1
    test(model, test_iter, threshold=threshold, k=k)


def test(model, data_iter, device='cpu', threshold=0.5, k=1):
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_p_at_k, test_loss = evaluate(model, data_iter, device=device, threshold=threshold, k=k)
    msg = 'Test Loss: {0:>5.2},  Test P@k: {1:>6.2%}'
    print(msg.format(test_loss, test_p_at_k))
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)


def evaluate(model, data_iter, device='cpu', threshold=0.5, k=1):
    model.eval()
    loss_total = 0
    p_at_k_total = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in data_iter:
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = F.cross_entropy(outputs, labels, reduction='mean')
            loss_total += loss
            p_at_k_total += batch_p_at_k_sum(outputs, labels, threshold=threshold, k=k)
            count += len(outputs)
    p_at_k = p_at_k_total / count
    loss = loss_total / count
    return p_at_k, loss


def batch_p_at_k_sum(outputs, labels, threshold=0.5, k=1):
    assert len(outputs) == len(labels), 'Different number of items, cannot compute P@k!'
    if len(outputs) == 0:
        return 0.5
    p_at_k_total = 0.
    topk_values, topk_indices = torch.topk(outputs, k, dim=1)
    topk_chose = (topk_values >= threshold).type(torch.int)
    label_indices = torch.arange(labels.shape[0]).unsqueeze(1).repeat((1, k))
    topk_labels = labels[[label_indices, topk_indices]].type(torch.int)
    num_p = topk_chose.sum(dim=1)
    num_tp = (topk_chose + topk_labels == 2).type(torch.int).sum(dim=1)

    for i, j in zip(num_tp, num_p):
        if j == 0:
            p_at_k_total += 0.5
        else:
            p_at_k_total += i / j
    return p_at_k_total


def get_time_diff(start_time):
    cur_time = time.time()
    time_diff = cur_time - start_time
    return time.strftime('%H:%M:%S', time.gmtime(time_diff))
