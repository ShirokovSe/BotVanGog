from aiogram import Bot, types
import asyncio
from aiogram.types import InputFile, ContentType, base, \
KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardMarkup, \
InlineKeyboardButton, CallbackQuery, ReplyKeyboardRemove
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import torch 
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
from torchvision.models import vgg19,efficientnet_b7, resnet50
from PIL import Image
import os
import numpy as np
import pickle
from style_transfer import find_mean_std, AdaIN, Decoder, Net
from conditional_GAN import Generator

#настройка webhook
WH_HOST = '18.195.213.155'
WH_PORT = 443
WH_PATH = "/"
WH_URL_BASE = f'https://{WH_HOST}{WH_PATH}'
WEBHOOK_SSL_CERT = './cert/webhook_cert.pem'  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = './cert/webhook_pkey.pem'  # Path to the ssl private key
#сервер
WB_HOST = 'localhost'
WB_PORT =  8569

#запуск webhook
async def on_startup(dp):
    await bot.set_webhook(WH_URL_BASE, certificate=open(WEBHOOK_SSL_CERT, 'rb'))


#инициализация переменных для нейронных сетей
DEVICE = torch.device("cpu")
NUM_CLASSES_GENERATE = 27
IMAGE_SIZE = 64
NUM_CHANNELS = 3
NOISE_SIZE = 150
FEATURE_MAP_GEN = 64

#инициализация переменных для бота
TOKEN = '5017870135:AAEn34eZMlsbFInerN0Dd66Kd3ilkrvVU6k'
PATH = './'
USER_PATH = './users/'
bot = Bot(token = TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage = storage)

#модель для переноса стиля
decoder = Decoder().to(DEVICE)
net = Net(decoder).to(DEVICE)
decoder.load_state_dict(torch.load(os.path.join(PATH,'good.w'), map_location = torch.device('cpu')))

#модель для генерации изображений
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(os.path.join(PATH,'generator_ab_last.w'), map_location = torch.device('cpu')))

#модель для классификации зданий
model_builds = resnet50(pretrained = True).to(DEVICE)
model_builds.fc = nn.Linear(in_features = 2048, out_features = 10, bias = True)
model_builds.load_state_dict(torch.load(os.path.join(PATH,'buildings.w'), map_location = torch.device('cpu')))

#модель для классификации изображений
model_images = efficientnet_b7(pretrained = True).to(DEVICE)
model_images.classifier = nn.Sequential(
    nn.Dropout(p = 0.5, inplace = True),
    nn.Linear(in_features = 2560, out_features = 10, bias = True)
  )
model_images.load_state_dict(torch.load(os.path.join(PATH,'images.w'), map_location=torch.device('cpu')))


#функция для генерации изображений
async def generate(DIR,label):
    noise = torch.randn((9,NOISE_SIZE,1,1), device = DEVICE)
    label = torch.tensor(np.repeat(label,9), device = DEVICE)
    generator.eval()
    with torch.no_grad():
        result = generator(noise, label)

    image = Image.fromarray((np.transpose(vutils.make_grid(result.detach(), padding=1, nrow = 3, normalize=True).cpu().numpy(),(1,2,0)) * 255).astype(np.uint8))
    image.save(os.path.join(DIR,'generated.png'))
    path = os.path.join(DIR,'generated.png')
    return path

#функция для переноса стиля
async def transfer(DIR):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512,512)),
        torchvision.transforms.ToTensor() ])

    test_content_img = Image.open(os.path.join(DIR,'content.png')).convert('RGB')
    ground_width, ground_len = test_content_img.size
    test_content_img = transforms(test_content_img)
    test_content_img = torch.unsqueeze(test_content_img, 0).to(DEVICE)

    test_style_img = Image.open(os.path.join(DIR,'style.png')).convert('RGB')
    test_style_img = transforms(test_style_img)
    test_style_img = torch.unsqueeze(test_style_img, 0).to(DEVICE)

    back_transform = torchvision.transforms.Resize((ground_len, ground_width))

    alpha = 1

    with torch.no_grad():
      content = test_content_img.to(DEVICE)
      style = test_style_img.to(DEVICE)
      content_f = net.encode(content)
      style_f = net.encode(style)
      normalized = AdaIN(content_f, style_f)
      normalized = (1.0 - alpha) * content_f + alpha * normalized
      output = decoder(normalized)

    output = back_transform(output.clone().detach())
    output = output.to("cpu").numpy()
    output = (output[0].transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(output)
    out.save(os.path.join(DIR, 'transform.png'))
    path = os.path.join(DIR,'transform.png')
    return path

#функция для классификации зданий
async def classify_buildings(DIR):
    transforms = torchvision.transforms.Compose([
        # smaller dimention of the image is matched to 512px while keeping the aspect ratio.
        torchvision.transforms.Resize(512),
        torchvision.transforms.ToTensor()
    ])
    test_content_img = Image.open(os.path.join(DIR,'classify_build.png')).convert('RGB')
    image = transforms(test_content_img)
    model_builds.eval()
    with torch.no_grad():
        result = torch.argmax(model_builds(image.unsqueeze(0)))
        result = result.detach().cpu().numpy()

    with open(os.path.join(PATH,'labels_buildings.pkl'),'rb') as file:
        label_enc = pickle.load(file)

    return label_enc.inverse_transform(np.array(result).reshape(1,)).item()

#функция для классификации изображений
async def classify_painting(DIR):
    transforms = torchvision.transforms.Compose([
        # smaller dimention of the image is matched to 512px while keeping the aspect ratio.
        torchvision.transforms.Resize(512),
        torchvision.transforms.ToTensor()
    ])
    test_content_img = Image.open(os.path.join(DIR,'classify_painting.png')).convert('RGB')

    image = transforms(test_content_img)
    model_images.eval()
    with torch.no_grad():
        result = torch.argmax(model_images(image.unsqueeze(0)))
        result = result.detach().cpu().numpy()

    with open(os.path.join(PATH,'labels_images.pkl'),'rb') as file:
        label_enc = pickle.load(file)

    return label_enc.inverse_transform(np.array(result).reshape(1,)).item()

#инициализация машины состояний бота
class FSMAdmin(StatesGroup):
    menu = State()
    style = State()
    content = State()
    classification_building = State()
    classification_images = State()
    generation = State()

#заводим словарь со всем состояниями для возможности вернуться в основное меню из любого состояния
all_states = [FSMAdmin.menu, FSMAdmin.style,FSMAdmin.content, FSMAdmin.classification_building,\
FSMAdmin.classification_images,FSMAdmin.generation]

#инициализация клавиатуры главного меню
button_menu_1 = KeyboardButton('Перенос стиля')
button_menu_2 = KeyboardButton('Классификация зданий')
button_menu_3 = KeyboardButton('Классификация картин')
button_menu_4 = KeyboardButton('Генерация')
menu_kb = ReplyKeyboardMarkup(resize_keyboard=True)
menu_kb.add(button_menu_1).add(button_menu_2).add(button_menu_3).add(button_menu_4)

#инициализаци кнопки выхода в главное меню
key_b_exit = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton('Назад'))

#инициализация клавиатуры для генерации изображений
button_gen_1 = KeyboardButton('Абстракционизм')
button_gen_2 = KeyboardButton('Экспрессионизм')
button_gen_3 = KeyboardButton('Минимализм')
button_gen_4 = KeyboardButton('Кубизм')
button_gen_5 = KeyboardButton('Поп Арт')
button_gen_6 = KeyboardButton('Романтизм')
button_gen_7 = KeyboardButton('Реализм')
button_gen_8 = KeyboardButton('Ренессанс')
button_gen_9 = KeyboardButton('Символизм')
button_gen_back = KeyboardButton('Назад')
generator_kb = ReplyKeyboardMarkup(resize_keyboard=True)
generator_kb.add(button_gen_1,button_gen_2,button_gen_3).add(button_gen_4,button_gen_5,button_gen_6).add(button_gen_7,button_gen_8,button_gen_9).add(button_gen_back)

#инициализация меню с командами
async def set_default_commands(dp):
    await dp.bot.set_my_commands([
        types.BotCommand("help", "Помощь"),
        types.BotCommand("restart", "Запустить бота заново"),
       
    ], scope=types.BotCommandScopeDefault())


@dp.message_handler(commands= ['start'])
async def process_start_command(msg: types.Message):
    await bot.send_message(msg.from_user.id,f'Привет, {msg.from_user.first_name}!\nЯ бот, который обожает искусство!\nИ вот, что я умею:\nПереносить стиль с одной картинки на другую\n\
Распознавать стили зданий\nРаспознавать стили картин\nРисовать новые картины в различных стилях', reply_markup = menu_kb)
    if not os.path.exists(os.path.join(USER_PATH,str(msg.from_user.id))):
        os.mkdir(os.path.join(USER_PATH,str(msg.from_user.id)))
    else:
        pass
    await FSMAdmin.menu.set()

@dp.message_handler(commands= ['restart'], state = all_states)
async def process_start_command(msg: types.Message, state: FSMContext):
    await FSMAdmin.menu.set()
    await msg.reply(f'Что-то пошло не так, но я в строю, все хорошо! Выбирай, что будем дальше делать', reply_markup = menu_kb)
    if not os.path.exists(os.path.join(USER_PATH,str(msg.from_user.id))):
        os.mkdir(os.path.join(USER_PATH,str(msg.from_user.id)))
    else:
        pass

@dp.message_handler(commands=['help'], state = all_states)
async def process_help_command(msg: types.Message):
    await FSMAdmin.menu.set()
    text = 'Я делаю разные интересные вещи, давай я тебе о них подробнее расскажу!\n\
Начну с переноса стиля, тебе необходимо следовать моим инструкциям. Но я повторюсь,\
сначала тебе нужно выбрать этот раздел, после моего сообщения прислать мне фотографию стиля, который ты хочешь, чтобы я применил, \
после отправки стиля, дождись моего подтверждения и пришли мне фотографию, на которую я буду наносить стиль, осталось немного подождать и получить результат.\
\n\nКлассификации работают идентично, тебе нужно выбрать раздел, дождаться моего сообщения, прислать мне фотографию и получить результат.\
\n\nГенерация картин, тольо представь, этих картин не существует в природе, все они сгенерированы нейронной сетью! Тебе следует выбрать этот раздел\
, а затем выбирать стили на твой вкус!\n\nЕсли что-то случилось с ботом, введи команду /restart.\nЕсли с ботом совсем все плохо, напиши этому парню: @shirsergey'
    await bot.send_message(msg.from_user.id,text)

@dp.message_handler(text = 'Назад',state = all_states)
async def menu(msg: types.Message, state: FSMContext):
    text = np.random.choice(['Что дальше?','Какие планы?','Чего изволите?','Чем теперь займемся?'])
    await msg.reply(text, reply_markup = menu_kb)
    await FSMAdmin.menu.set()

@dp.message_handler(text = 'Перенос стиля', state = FSMAdmin.menu)
async def style_transfer(msg: types.Message, state: FSMContext):
    
    await msg.reply('Здорово, теперь приготовься прислать мне две фотограии: первая фотография будет стиль, а вторая любая другая фотография на твой выбор\
\nПришли сначала мне фотографию стиля', reply_markup= key_b_exit)

    await FSMAdmin.style.set()


@dp.message_handler(content_types= [ContentType.PHOTO, ContentType.DOCUMENT], state = [FSMAdmin.style, FSMAdmin.menu])
async def images(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/style.png')

    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/style.png')

    await FSMAdmin.content.set()
    await msg.reply('Отлично, теперь пришли мне твою фотографию, чтобы я мог ее разукрасить', reply_markup= key_b_exit)



@dp.message_handler(content_types= [ContentType.PHOTO, ContentType.DOCUMENT], state = [FSMAdmin.content,FSMAdmin.menu])
async def images(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/content.png')
    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/content.png')
  
    await bot.send_message(msg.from_user.id, text = 'Супер, примерно 10 секунд, и твоя фотография будет готова!', reply_markup = ReplyKeyboardRemove())

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))

    transf = await transfer(DIR)
    st_tr = InputFile(transf)
    await bot.send_photo(msg.from_user.id, st_tr)
    await FSMAdmin.menu.set()
    await bot.send_message(msg.from_user.id, text = 'Чем дальше займемся?',reply_markup = menu_kb)

@dp.message_handler(text = 'Классификация зданий', state = FSMAdmin.menu)
async def classify_request(msg: types.Message, state: FSMContext):
    await bot.send_message(msg.from_user.id, text = "Я могу различать стили зданий, на сегодняшний день я знаю 5 основных стилей\n\
 Конечно, я могу ошибаться, потому что стиль здания не всегда может быть определен одназначно", reply_markup= key_b_exit)
    await FSMAdmin.classification_building.set()

@dp.message_handler(content_types = [ContentType.PHOTO, ContentType.DOCUMENT], state = [FSMAdmin.classification_building, FSMAdmin.menu])
async def classify_it(msg: types.Message, state: FSMContext):
    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/classify_build.png')

    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/classify_build.png')

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))
    await bot.send_message(msg.from_user.id,'Так, дай мне пару секунд')
    style = await classify_buildings(DIR)
    build_style = {
    'Modern':'модерн',
    'Barroco':'барроко',
    'Classic':'классицизм',
    'Gothic':'готический стиль',
    'Casual':'обычный жилой дом'
    }
    await bot.send_message(msg.from_user.id,text = f"Это {build_style[style]}")
    await FSMAdmin.menu.set()
    await bot.send_message(msg.from_user.id, text = "Что теперь выберешь?",reply_markup = menu_kb)

@dp.message_handler(text = 'Классификация картин', state = FSMAdmin.menu)
async def classify_request(msg: types.Message, state: FSMContext):
    await bot.send_message(msg.from_user.id, text = "Отличный выбор! Я знаю 10 основных стилей в \
живописи и постарасюь определить его для тебя\nКонечно, я могу делать ошибки, ведь живопись достаточно\
 неоднозначна и не всегда можно точно определить стиль", reply_markup= key_b_exit)
    await FSMAdmin.classification_images.set()


@dp.message_handler(content_types = [ContentType.PHOTO, ContentType.DOCUMENT], state = [FSMAdmin.classification_images, FSMAdmin.menu])
async def classify_it(msg: types.Message, state: FSMContext):

    if msg.content_type == ContentType.PHOTO:
        await msg.photo[-1].download(destination_file = USER_PATH + str(msg.from_user.id) + '/classify_painting.png')
    if msg.content_type == ContentType.DOCUMENT:
        file_id = msg.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, USER_PATH + str(msg.from_user.id) + '/classify_painting.png')

    DIR = os.path.join(USER_PATH,str(msg.from_user.id))
    await bot.send_message(msg.from_user.id,'Круто, дай мне пару секунд')
    style = await classify_painting(DIR)
    await bot.send_message(msg.from_user.id,text = f"Итак, стиль твоей картины {style}")
    await FSMAdmin.menu.set()
    await bot.send_message(msg.from_user.id, text = "Что теперь выберешь?",reply_markup = menu_kb)

@dp.message_handler(text = 'Генерация', state = FSMAdmin.menu)
async def classify_request(msg: types.Message, state: FSMContext):
    await bot.send_message(msg.from_user.id, text = "Я люблю искусство, поэтому рисую коллажи в различном стиле\n\
Выбери стиль и я пришлю тебе коллаж из 9 картин", reply_markup = generator_kb)
    await FSMAdmin.generation.set()


    
@dp.message_handler(text = ['Абстракционизм','Кубизм','Экспрессионизм','Минимализм','Поп Арт','Романтизм',\
'Реализм','Ренессанс','Символизм'],state = [FSMAdmin.generation, FSMAdmin.menu])
async def generate_style(msg: types.Message, state: FSMContext):

    labels_generate = {
        'Абстракционизм':0,
        'Экспрессионизм':9,
        'Минимализм':14,
        'Кубизм':7,
        'Поп Арт':19,
        'Романтизм':23,
        'Реализм':21,
        'Ренессанс':11,
        'Символизм':24
    }

    answer_generate = {
    'Абстракционизм':"Эх, абстракционизм... Малевич, Кандинский...",
    'Экспрессионизм':"Экспрессионизм, why not?",
    'Минимализм':"Меньше деталей, больше смысла",
    'Кубизм':"Ммм, кубизм, хороший выбор",
    'Поп Арт':"Поп Арт, Энди Уорхол и компания",
    'Романтизм':"Ох, уж этот романтизм, стольких свел с ума",
    'Реализм':"Однако, жизненно",
    'Ренессанс':"Когда-то было модно",
    'Символизм':"Как говорится, символично!"
    }
    label = labels_generate[msg.text]
    text = answer_generate[msg.text]
    DIR = os.path.join(USER_PATH,str(msg.from_user.id))
    await bot.send_message(msg.from_user.id, text, reply_markup = generator_kb)
    picture = InputFile(await generate(DIR, label))
    await bot.send_photo(msg.from_user.id, picture)


#все текстовые сообщения удаляются ботом, за исклчением текста кнопок и команд
@dp.message_handler(content_types=ContentType.TEXT, state = all_states)
async def delete_message(msg: types.Message, state: FSMContext):
    if msg.text in ['Классификация зданий','Классификация картин', 'Генерация','Перенос стиля','Абстракционизм','Кубизм','Экспрессионизм','Минимализм','Поп Арт','Романтизм','Реализм','Ренессанс','Символизм','/start','/help','restart','Назад']:
        pass
    else:
        await bot.delete_message(msg.from_user.id, msg.message_id)

#картинки, присланные в неподходящих разделах удаляются
@dp.message_handler(content_types=[ContentType.PHOTO, ContentType.DOCUMENT],state = [FSMAdmin.generation,FSMAdmin.menu])
async def delete_photo(msg: types.Message, state:FSMContext):
    await bot.delete_message(msg.from_user.id, msg.message_id)


#аудио и видео сообщения тоже удаляются
@dp.message_handler(content_types=[ContentType.AUDIO, ContentType.VOICE, ContentType.VIDEO_NOTE, ContentType.ANIMATION],state = all_states)
async def delete_photo(msg: types.Message, state:FSMContext):
    await bot.delete_message(msg.from_user.id, msg.message_id)


if __name__ == '__main__':
	executor.start_webhook(dispatcher=dp,webhook_path= WH_PATH, on_startup=on_startup,host=WB_HOST, port=WB_PORT)
