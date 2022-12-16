import torch.nn
import argparse
from torch.utils.data import DataLoader
from utils.datasets import Test_Dataset
from models.encoder_decoder import FED, INL
from utils.utils import *
from utils.metric import *

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='DECODE')
    parser.add_argument('--noise-type', '-n', default='JPEG', type=str, help='The noise type added to the watermarked images.')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='The batch size.')
    parser.add_argument('--messages-path', '-m', default="messages", type=str, help='The embedded messages')
    parser.add_argument('--watermarked-image', '-o', default="output_images", type=str, help='The watermarked images')

    args = parser.parse_args()

    inn_data = Test_Dataset(args.watermarked_image, "png")
    inn_loader = DataLoader(inn_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    error_history = []

    with torch.no_grad():
        if args.noise_type in ["JPEG", "HEAVY"]:
            fed_path = os.path.join("experiments",args.noise_type,"FED.pt")
            fed = FED().to(device)
            load(fed_path, fed)
            fed.eval()

            if args.noise_type == "HEAVY":
                inl_path = os.path.join("experiments", args.noise_type, "INL.pt")
                inl = INL().to(device)
                load(inl_path, inl)
                inl.eval()

            for idx, watermarked_images in enumerate(inn_loader):
                watermarked_images = watermarked_images.to(device)
                embedded_messgaes = torch.load(os.path.join(args.messages_path,"message_{}.pt".format(idx+1)))

                all_zero = torch.zeros(embedded_messgaes.shape).to(device)

                if args.noise_type == "HEAVY":
                    watermarked_images = inl(watermarked_images.clone(), rev=True)

                reversed_img, extracted_messages = fed([watermarked_images, all_zero], rev=True)

                error_rate = decoded_message_error_rate_batch(embedded_messgaes, extracted_messages)

                error_history.append(error_rate)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print('error : {:.3f}'.format(np.mean(error_history)))


if __name__ == '__main__':
    main()
