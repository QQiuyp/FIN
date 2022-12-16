import torch.nn
import argparse
from torch.utils.data import DataLoader
from utils.datasets import Test_Dataset
from models.encoder_decoder import FED, INL
from utils.utils import *
from utils.jpeg import JpegTest
from utils.metric import *
import torchvision

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='ENCODE')
    parser.add_argument('--noise-type', '-n', default='JPEG', type=str, help='The noise type will be added to the watermarked images.')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', default="test_images", type=str, help='The images to watermark')
    parser.add_argument('--source-image-type', '-t', default="png", type=str, help='The type of the input images')
    parser.add_argument('--messages-path', '-m', default="messages", type=str, help='The messages to embed')
    parser.add_argument('--watermarked-image', '-o', default="output_images", type=str, help='The output images')

    args = parser.parse_args()

    inn_data = Test_Dataset(args.source_image, args.source_image_type)
    inn_loader = DataLoader(inn_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    psnr_history = []

    with torch.no_grad():
        if args.noise_type in ["JPEG", "HEAVY"]:
            if args.noise_type == "JPEG":
                noise_layer = JpegTest(50)
            fed_path = os.path.join("experiments",args.noise_type,"FED.pt")
            fed = FED().to(device)
            load(fed_path, fed)
            fed.eval()
            for idx, source_images in enumerate(inn_loader):
                source_images = source_images.to(device)
                source_messgaes = torch.Tensor(np.random.choice([-0.5, 0.5], (source_images.shape[0], 64))).to(device)

                stego_images, left_noise = fed([source_images, source_messgaes])

                if args.noise_type == "JPEG":
                    final_images = noise_layer(stego_images.clone())
                else:
                    final_images = stego_images

                psnr_value = psnr(source_images, stego_images, 255)

                for i in range(source_images.shape[0]):
                    number = 1 + i + idx * source_images.shape[0]
                    torchvision.utils.save_image(((final_images[i] / 2) + 0.5),
                                                 os.path.join(args.watermarked_image,"{}.png".format(number)))

                torch.save(source_messgaes, os.path.join(args.messages_path,"message_{}.pt".format(idx+1)))

                psnr_history.append(psnr_value)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print('psnr: {:.3f}'.format(np.mean(psnr_history)))



if __name__ == '__main__':
    main()
