import torch.nn
from utils.utils import *
from utils.metric import *
from utils.datasets import *
from models.encoder_decoder import FED
from utils.jpeg import JpegSS, JpegTest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def stego_loss_fn(stego, cover):
    loss_fn = torch.nn.MSELoss(reduce=True)
    loss = loss_fn(stego, cover)
    return loss.to(device)

def message_loss_fn(recover_message, message):
    loss_fn = torch.nn.MSELoss(reduce=True)
    loss = loss_fn(recover_message, message)
    return loss.to(device)

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)

fed = FED()
fed.cuda()
params_trainable = (list(filter(lambda p: p.requires_grad, fed.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

if c.train_continue:
    load(fed, c.MODEL_PATH + c.suffix)

setup_logger('train', 'logging', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')

noise_layer = JpegSS(50)
test_noise_layer = JpegTest(50)

for i_epoch in range(c.epochs):

    loss_history = []
    stego_loss_history = []
    message_loss_history = []
    stego_psnr_history = []
    error_history = []

    #################
    #     train:    #
    #################

    fed.train()
    for idx_batch, cover_img in enumerate(trainloader):
        cover_img = cover_img.to(device)

        message = torch.Tensor(np.random.choice([-0.5, 0.5], (cover_img.shape[0], c.message_length))).cuda()
        input_data = [cover_img, message]

        #################
        #    forward:   #
        #################

        stego_img, left_noise = fed(input_data)
        stego_noise_img = noise_layer(stego_img.clone())

        #################
        #   backward:   #
        ################

        guass_noise = torch.zeros(left_noise.shape).cuda()
        output_data = [stego_noise_img, guass_noise]
        re_img, re_message = fed(output_data, rev=True)

        stego_loss = stego_loss_fn(stego_img, cover_img)
        message_loss = message_loss_fn(re_message, message)

        total_loss = c.message_weight * message_loss + c.stego_weight * stego_loss
        total_loss.backward()

        optim.step()
        optim.zero_grad()

        psnr_temp_stego = psnr(cover_img, stego_img, 255)

        error_rate = decoded_message_error_rate_batch(message, re_message)

        loss_history.append([total_loss.item(), 0.])
        stego_loss_history.append([stego_loss.item(), 0.])
        message_loss_history.append([message_loss.item(), 0.])
        stego_psnr_history.append([psnr_temp_stego, 0.])
        error_history.append([error_rate, 0.])

    epoch_losses = np.mean(np.array(loss_history), axis=0)
    stego_epoch_losses = np.mean(np.array(stego_loss_history), axis=0)
    message_epoch_losses = np.mean(np.array(message_loss_history), axis=0)
    stego_psnr = np.mean(np.array(stego_psnr_history), axis=0)
    error = np.mean(np.array(error_history), axis=0)

    epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

    logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
    logger_train.info(
        f"Train epoch {i_epoch}:   "
        f'Loss: {epoch_losses[0].item():.4f} | '
        f'Stego_Loss: {stego_epoch_losses[0].item():.4f} | '
        f'Message_Loss: {message_epoch_losses[0].item():.4f} | '
        f'Stego_Psnr: {stego_psnr[0].item():.4f} |'
        f'Error:{1 - error[0].item():.4f} |'
    )

    #################
    #     val:      #
    #################
    with torch.no_grad():
        stego_psnr_history = []
        error_history = []

        fed.eval()
        for test_cover_img in testloader:
            test_cover_img = test_cover_img.to(device)

            test_message = torch.Tensor(np.random.choice([-0.5, 0.5], (test_cover_img.shape[0], c.message_length))).to(device)

            test_input_data = [test_cover_img, test_message]

            #################
            #    forward:   #
            #################

            test_stego_img, test_left_noise = fed(test_input_data)

            if c.noise_flag:
                test_stego_noise_img = test_noise_layer(test_stego_img.clone())

            #################
            #   backward:   #
            #################

            test_z_guass_noise = torch.zeros(test_left_noise.shape).cuda()

            test_output_data = [test_stego_noise_img, test_z_guass_noise]

            test_re_img, test_re_message = fed(test_output_data, rev=True)


            psnr_temp_stego = psnr(test_cover_img, test_stego_img, 255)
            psnr_temp_recover = psnr(test_cover_img, test_re_img, 255)

            error_rate = decoded_message_error_rate_batch(test_message, test_re_message)

            stego_psnr_history.append(psnr_temp_stego)
            error_history.append(error_rate)

        logger_train.info(
            f"TEST:   "
            f'PSNR_STEGO: {np.mean(stego_psnr_history):.4f} | '
            f'Error: {1 - np.mean(error_history):.4f} | '
        )

    if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
        torch.save({'opt': optim.state_dict(),
                    'net': fed.state_dict()},
                   c.MODEL_PATH + 'fed_' + str(np.mean(stego_psnr_history)) + '_%.5i' % i_epoch + '.pt')


torch.save({'opt': optim.state_dict(),
            'net': fed.state_dict()},
           c.MODEL_PATH + 'fed' + '.pt')
