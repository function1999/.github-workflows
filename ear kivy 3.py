import tensorflow
from PIL import Image
from PIL import ImageOps
import numpy as np
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window

fontname = "H2GTRM.TTF"
Window.clearcolor = (1, 1, 1, 1)

class EarPhysiognomyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # 업로드 버튼
        self.upload_button = Button(text='사진 업로드', size_hint=(1, None), font_name=fontname)
        self.upload_button.bind(on_release=self.show_file_chooser)
        
        # 결과 표시 버튼
        self.result_button = Button(text='귀 관상 결과 표시', size_hint=(1, None), font_name=fontname)
        self.result_button.bind(on_release=self.show_result)
        
        # 이미지 표시
        self.image = KivyImage(source='', size_hint=(1, None), height=400)
        
        # 결과 레이블
        self.result_label = Label(text='', size_hint=(1, None), height=30, font_name=fontname)
        
        self.layout.add_widget(self.image)
        self.layout.add_widget(self.upload_button)
        self.layout.add_widget(self.result_button)
        self.layout.add_widget(self.result_label)
        
        return self.layout
    
    def close_popup(self, instance):
        Window.close()
    
    def show_file_chooser(self, instance):
        self.file_chooser = FileChooserListView(font_name=fontname)
        self.file_chooser.bind(on_submit=self.load_image)
        close_button = Button(text='닫기', size_hint=(1, None), font_name=fontname)

        popup = Popup(title='file chooser', content=self.file_chooser, size_hint=(0.9, 0.9))
        popup.open(on_release=self.close_popup)
        self.file_chooser.add_widget(close_button)

        close_button.bind(on_press = popup.dismiss)
    


    
    def load_image(self, instance, selection, touch):
        if selection:
            selected_file = selection[0]
            self.image.source = selected_file
            self.result_label.text = ''
    
    def show_result(self, instance):
        if self.image.source:
            image_path = self.image.source
            model_path = os.path.join("converted_keras", "keras_model.h5")
            labels_path = os.path.join("converted_keras", "labels.txt")

            # Load the model
            model = tensorflow.keras.models.load_model(model_path)

            # Prepare the image for prediction
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make a prediction
            prediction = model.predict(data)

            # Load class labels
            with open(labels_path, 'rt', encoding='UTF-8') as f:
                readLines = f.readlines()

            # Extract prediction probabilities for each class
            probability_class_0 = prediction[0, 0]
            probability_class_1 = prediction[0, 1]
            probability_class_2 = prediction[0, 2]
            probability_class_3 = prediction[0, 3]
            probability_class_4 = prediction[0, 4]

            # 예측 확률과 레이블을 리스트로 묶어서 저장
            predictions = [(probability_class_0, readLines[0]), (probability_class_1, readLines[1]), (probability_class_2, readLines[2]), (probability_class_3, readLines[3]), (probability_class_4, readLines[4])]

            # 확률이 높은 순서로 정렬
            predictions.sort(reverse=True)

            # 상위 2개 예측을 출력
            result_text = "Top 2 Predictions:\n"
            for i in range(2):
                probability, label = predictions[i]
                result_text += f" {label} : {probability * 100:.2f} %\n"

            # 팝업 창을 생성하고 레이블 위젯을 추가
            close_button = Button(text='닫기', size_hint=(1, None), font_name=fontname)
            popup = Popup(title='result', size_hint=(None, None), size=(800, 400))
            popup_label = Label(text=result_text, font_name=fontname, font_size='25sp', halign='left', valign='top', markup=True)
            popup_label.bind(size=popup_label.setter('text_size'))
            popup.content = popup_label  # 레이블을 팝업창의 콘텐츠로 설정
            popup.open()
            popup.content.add_widget(close_button)
        
        else:
            self.result_label.text = "Please upload an image first."
       

if __name__ == '__main__':
    EarPhysiognomyApp().run()