import torch
import torch_dipu
import torch.nn.functional as F

log_probs = torch.randn(50, 16, 20).log_softmax(2).detach()
log_probs_device = log_probs.cuda()
log_probs.requires_grad_()
log_probs_device.requires_grad_()

targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)

loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()


loss1 = F.ctc_loss(log_probs_device, targets.cuda(), input_lengths.cuda(), target_lengths.cuda())
loss1.backward()

assert torch.allclose(loss1.cpu(), loss, atol = 1e-3, rtol = 1e-3)
assert torch.allclose(log_probs_device.grad.cpu(), log_probs.grad, atol = 1e-3, rtol = 1e-3)



log_probs = torch.randn(50, 16, 20).log_softmax(2).detach()
log_probs_device = log_probs.cuda()
log_probs.requires_grad_()
log_probs_device.requires_grad_()

targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)

loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction = 'none')
loss.backward(torch.ones_like(loss))

loss1 = F.ctc_loss(log_probs_device, targets.cuda(), input_lengths.cuda(), target_lengths.cuda(), reduction = 'none')
loss1.backward(torch.ones_like(loss1))

assert torch.allclose(loss1.cpu(), loss, atol = 1e-3, rtol = 1e-3)
assert torch.allclose(log_probs_device.grad.cpu(), log_probs.grad, atol = 1e-3, rtol = 1e-3)