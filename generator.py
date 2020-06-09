import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

Parameter_list = {"background_path": "Project_doc/background/",  # 生成数据的背景路径
                  "image_width": 280,
                  "image_height": 32,
                  "font_path": "Project_doc/font/",  # 字体文件路径
                  "corpus_path": "Project_doc/info.txt",  # 语料库路径, 用于生成数据的文本语料库
                  "dict_path": "Project_doc/dict.txt",  # 字典路径
                  "len_word": 10,  # 生成词长度
                  "output_label_path": "train_data.txt",  # 输出label文件
                  "output_data_path": "train_data_img/",  # 输出data路径
                  }


class Generator_img():
    '''
    用于生成训练数据的图片 和 标签
    '''

    def __init__(self, Parameter_list):
        # 生成参数
        self.generative_word_length = Parameter_list["len_word"]  # 生成词的长度
        # 背景图参数
        self.bground_path = Parameter_list["background_path"]
        self.bg_width = Parameter_list["image_width"]
        self.bg_height = Parameter_list["image_height"]
        # 字体参数
        self.font_path = Parameter_list["font_path"]
        # 语料库参数
        self.corpus_path = Parameter_list["corpus_path"]
        # load corpus
        self.corpus_str = self.load_corpus()

    def load_corpus(self):
        '''
        处理具有工商信息语义信息的语料库，去除空格等不必要符号
        :return: 返回加载后的语料库字符串
        '''
        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            info_list = [part.strip().replace('\t', '') for part in file.readlines()]
            info_str = ''.join(info_list)
        return info_str

    def generate_word_from_info_str(self, quantity=10):
        '''
        从文字库中随机选择n个字符
        :param quantity: 随机选择字符的数量
        :return:
        '''
        start = random.randint(0, len(self.corpus_str) - 11)
        end = start + quantity
        random_word = self.corpus_str[start:end]
        return random_word

    def random_word_color(self):
        '''
        随机选择字体颜色
        :return:
        '''
        font_color_choice = [[54, 54, 54], [54, 54, 54], [105, 105, 105]]
        font_color = random.choice(font_color_choice)
        noise = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
        font_color = (np.array(font_color) + noise).tolist()
        # print('font_color：',font_color)
        return tuple(font_color)

    def generate_img_bground(self):
        '''
        根据背景图 图片的宽 图片的高 生成一张图片
        :return:
        '''
        bground_list = os.listdir(self.bground_path)
        bground_choice = random.choice(bground_list)
        bground = Image.open(self.bground_path + bground_choice)
        x, y = random.randint(0, bground.size[0] - self.bg_width), random.randint(0, bground.size[1] - self.bg_height)
        bground = bground.crop((x, y, x + self.bg_width, y + self.bg_height))
        return bground

    def darken_func(self, image):
        '''
        输入一张图片 模糊后输出
        :param image: input image
        :return: after image
        '''
        filter_ = random.choice(
            [ImageFilter.SMOOTH,
             ImageFilter.SMOOTH_MORE,
             ImageFilter.GaussianBlur(radius=1.3)]
        )
        image = image.filter(filter_)
        return image

    def random_x_y(self, bground_size, font_size):
        '''
        随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
        :param bground_size:
        :param font_size:
        :return:
        '''
        width, height = bground_size
        # print(bground_size)
        # 为防止文字溢出图片，x，y要预留宽高
        x = random.randint(0, width - font_size * 10)
        y = random.randint(0, int((height - font_size) / 4))
        return x, y

    def random_font_size(self):
        '''随机选择字体大小'''
        font_size = random.randint(24, 27)
        return font_size

    def random_font(self):
        '''随机选择字体包'''
        font_list = os.listdir(self.font_path)
        random_font = random.choice(font_list)
        return self.font_path + random_font

    def random_choice_in_process_func(self):
        '''选取作用函数'''
        pass

    def rotate_func(self):
        '''旋转函数'''
        pass

    def random_noise_func(self):
        '''噪声函数'''
        pass

    def stretching_func(self):
        '''字体拉伸函数'''
        pass

    def generate_main(self):
        '''
        生成数据的主函数
        :return: 返回一个PIL的img对象 和img中的文本.
        '''
        # 生成图片中的文本
        generate_word = self.generate_word_from_info_str(self.generative_word_length)

        # 生成一张背景图片
        generate_image = self.generate_img_bground()
        # 随机选取字体大小
        font_size = self.random_font_size()
        # 随机选择字体
        font = self.random_font()
        # 随机选取文字贴合的坐标 x,y
        draw_x, draw_y = self.random_x_y(generate_image.size, font_size)

        # 将文本贴到背景图片
        font = ImageFont.truetype(font, font_size)
        draw = ImageDraw.Draw(generate_image)
        draw.text((draw_x, draw_y), generate_word, fill=self.random_word_color(), font=font)

        # 随机选取作用函数和数量作用于图片
        # random_choice_in_process_func()
        generate_image = self.darken_func(generate_image)
        # generate_image = generate_image.rotate(0.3)
        return generate_image, generate_word

    def show_generate_data(self):
        '''显示生成一张的数据情况'''
        img, label = self.generate_main()
        img.show()
        print(label)


class Generate_train():
    '''
    用于创建一个生成器 生成的数据直接对接模型并训练模型.
    '''
    def __init__(self, Parameter_list):
        self.gen = Generator_img(Parameter_list)
        self.dict_path = Parameter_list["dict_path"]
        self.transformer_dict = self.load_dict()

    def load_dict(self):
        '''
        加载字典
        :param dict_path: input dict path
        :return: dict
        '''
        lang_dict = {}
        with open(self.dict_path) as d:
            for i, j in enumerate(d.readlines()):
                lang_dict[j.strip()] = (i)
            # lang_dict = d.readlines()
        return lang_dict

    def transformer_data(self, input_data):
        '''
        传入需要转换的文件　转换后的文本写入本地
        :param input_file: input_transformer_data
        :return: output data
        '''
        data, label = input_data
        label = list(label)
        label = list(map(lambda x: str(self.transformer_dict[x]), label))
        output_label = " ".join(label)
        output_data = np.array(data)
        return output_data, output_label

    def batch_generator(self, batch_size=128, max_label_len=10):
        '''
        训练数据生成器
        :param batch_size: 批次大小
        :param max_label_len: label的最大长度
        :return:
        '''
        while True:
            datas = []
            labels = []
            for _ in range(batch_size):
                data, label = self.gen.generate_main()
                data, label = self.transformer_data([data, label])  # 此处转换生成的数据
                datas.append(data)
                labels.append(label)
            yield np.array(datas), np.array(labels)


class Save_train_data():
    '''
    用于生成数据 并保存到本地用于训练 ocr 的识别模型
    '''
    def __init__(self, Parameter_list):
        self.gen = Generator_img(Parameter_list)
        self.output_label_path = Parameter_list["output_label_path"]
        self.output_data_path = Parameter_list["output_data_path"]
        if not os.path.exists(self.output_data_path):
            os.mkdir(self.output_data_path)
        if os.path.exists(self.output_label_path):
            os.remove(self.output_label_path)

    def gen_and_save_data(self, save_num=100):
        '''
        生成数据并保存到本地
        :param save_num: 生成数据的数量
        :return:
        '''
        from tqdm import tqdm
        start_img_name = 10000000
        label_file = open(self.output_label_path, "a+")
        for _ in tqdm(range(save_num)):
            start_img_name += 1
            img, text = self.gen.generate_main()
            label_file.write(str(start_img_name) + '.png ' + text + '\n')
            img.save(self.output_data_path + str(start_img_name) + '.png')
        label_file.close()


if __name__ == '__main__':
    # 生成一张图片看效果
    gen_data = Generator_img(Parameter_list)
    gen_data.show_generate_data()
    # # 保存训练数据到本地
    # save = Save_train_data(Parameter_list)
    # save.gen_and_save_data(save_num=10)
    # # 创建一个生成器 可直接对接模型
    # gen_loader = Generate_train(Parameter_list)
    # loader = gen_loader.batch_generator(batch_size=128, max_label_len=10)