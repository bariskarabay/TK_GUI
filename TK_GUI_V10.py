#-*- coding: utf-8 -*-

from tkinter import *
from tkinter import messagebox
#from Tkinter import*
#from tkMessageBox import*
from fileinput import*
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import csv
from tkinter.filedialog import askopenfilename
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import os
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import IndexToString
from time import time
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.ml import Pipeline
import threading
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
os.environ["PYTHONIOENCODING"] = "utf-8";
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class kisi:
   def __init__(self): 
    self.isim=""
    self.soyisim=""
    self.ceptelno=""
    self.evtelno=""
    

class adresDefteri(kisi):
 def __init__(self):
  self.directory = 'sonuclar'
  self.createFolder()
  self.sc = SparkContext('local')
  spark = SparkSession(self.sc)
  spark = SparkSession \
                  .builder \
                  .appName("Python Spark Logistic Regression example") \
                  .config('spark.executor.heartbeatInterval', '3600s') \
                  .config("spark.some.config.option", "some-value") \
                  .getOrCreate()
  locale = self.sc._jvm.java.util.Locale
  locale.setDefault(locale.forLanguageTag("en-US"))
  
  self.catcols = ['targtype1_txt']
  self.num_cols = ['country', 'region','attacktype1','weaptype1']
  self.labelCol = 'gname'

  Root=Tk()
  Root.geometry("800x600")
  Root.title("Yaşanan Terör Olaylarını İçeren Büyük Verinin Makine Öğrenmesi Teknikleri İle Analizi")

  menu = Menu(Root)
  filemenu = Menu(menu)
  menu.add_cascade(label="File", menu=filemenu)
  filemenu.add_command(label="CSV View", command=self.secVeGoster)
  filemenu.add_separator()
  filemenu.add_command(label="Çıkış", command=Root.quit)
  
  filemenu.add_separator()
  filemenu.add_command(label="Yeniden Başlat", command=self.restart_program)

  helpmenu = Menu(menu)
  menu.add_cascade(label="Yardım", menu=helpmenu)
  helpmenu.add_command(label="Hakkında...", command=self.Hakkinda)

  
  Root.configure(background='yellow',menu=menu)
  global HakkindaPencere,combo
  self.nameText = StringVar()
  self.selected1 = IntVar()
  self.selected1.set(1)
  self.selected2 = IntVar()
  self.selected2.set(3)

  
  self.egitimLbl=Label(text="  Eğitim Verisi",width=30,height=3,fg="red",bg="yellow")
  self.egitimLbl.grid(row=0,column=0)

  self.egitimTxt=Entry(textvariable = self.nameText, fg="red",bg="yellow")
  self.egitimTxt.grid(row=0,column=1)

  self.egitimSec=Button(text="  ...  ",command=self.secim,width=10,height=1,fg="red",bg="yellow")
  self.egitimSec.grid(row=0,column=2)

  self.dataSayisiLbl=Label(text="  Data Sayısı Girin (Maks:181600)",width=30,height=3,fg="red",bg="yellow")
  self.dataSayisiLbl.grid(row=0,column=3)

  self.dataSayisiTxt=Entry(fg="red",bg="yellow")
  self.dataSayisiTxt.grid(row=0,column=4)

  self.testLbl=Label(text="  Test Verisi Oranı %",width=30,height=3,fg="red",bg="yellow")
  self.testLbl.grid(row=1,column=0)

  self.testTxt=Entry(fg="red",bg="yellow")
  self.testTxt.grid(row=1,column=1)  

  self.algoritmaLbl=Label(text="  Algoritma Seçiniz:",width=30,height=3,fg="red",bg="yellow")
  self.algoritmaLbl.grid(row=2,column=0)


  self.rad1 = Radiobutton(text='Hepsini karşılaştır',variable=self.selected1, value=1,command=self.secilenRadio1)
  self.rad1.grid(column=1, row=2)

  self.rad2 = Radiobutton(text='Bir Algoritma Seçiniz:',variable=self.selected1, value=2,command=self.secilenRadio1)
  self.rad2.grid(column=2, row=2)
  
  self.combo = ttk.Combobox (Root, state='readonly')
  self.combo['values']= ("Logistic Regression", "Naive Bayes", "Random Forest Classifier", "Decision Tree Classifier","Support Vector Machine","KNN" )
  #self.combo.current(-1) #set the selected item
  #self.combo.grid(column=3, row=2)
  

  self.ulkeLbl=Label(text="  Ülke Seçiniz:",width=30,height=3,fg="red",bg="yellow")
  self.ulkeLbl.grid(row=3,column=0)

  self.rad3 = Radiobutton(text='Tüm Ülkeler İçin', variable=self.selected2, value=3,command=self.secilenRadio2)
  self.rad3.grid(column=1, row=3)
  
  self.rad4 = Radiobutton(text='Ülke Seçin:', variable=self.selected2, value=4,command=self.secilenRadio2)
  self.rad4.grid(column=2, row=3)
  
  self.comboulke = ttk.Combobox (Root, state='readonly')
  self.comboulke['values']= ("Türkiye", "ABD", "İran", "Pakistan", "Irak","Afganistan","Suriye")
  #self.comboulke.grid(column=3, row=3)

  #209 Turkey
  #217 ABD
  #94 İran
  #153 Pakistan
  #95 Irak
  #4 Afganistan
  #200 Suriye

  #self.comboulke.current(1) #set the selected item
  #image=photo3, ekler , compound=LEFT resmi sola ceker

  self.YukleBtn=Button(text="Veriyi Yükle", command=self.secilenDosya,width=20,height=3,fg="red",bg="yellow")
  self.YukleBtn.grid(row=4,column=2)

  self.DonusumBtn=Button(text="  Dönüşümü Başlat  ", command=self.DonusumuBaslat,width=20,height=3,fg="red",bg="yellow")
  self.DonusumBtn.grid(row=5,column=2)

  self.ModelBtn=Button(text="  Modeli Eğit  ", command=self.modeliEgit,width=20,height=3,fg="red",bg="yellow")
  self.ModelBtn.grid(row=6,column=2)

  self.SonucBtn=Button(text="  Sonucu Göster  ", command=self.csvView,width=20,height=3,fg="red",bg="yellow")
  self.ExportCsvBtn=Button(text="  Export CSV  ", command=self.exportCSV,width=20,height=3,fg="red",bg="yellow")


  #self.listele=Button(text="Listele",command=self.listele,width=30,height=3,fg="red",bg="yellow")
  #self.listele.grid(row=7,column=0)



  
  mainloop()	
     

 def restart_program(self):
	 #os.execv(sys.executable, ['python'] + sys.argv)
     import _winapi
     x = _winapi.GetCurrentProcess()
     _winapi.ExitProcess(x)
	 
     #self.egitimTxt.delete(0, END)
     #self.dataSayisiTxt.delete(0, END)
     #self.comboulke.config(state=DISABLED)
	 #self.combo.config(state=DISABLED)
	 #self.YukleBtn.grid(row=4,column=2)
	 #self.DonusumBtn.grid(row=5,column=2)
	 #self.ModelBtn.grid(row=6,column=2)
	 #self.SonucBtn.grid_remove()
	 #self.ExportCsvBtn.grid_remove()
	

 def returnUlkeInt(self):
    self.comboUlkeDeger =self.comboulke.current()
    if self.comboUlkeDeger==0:
       return 209 
    elif self.comboUlkeDeger==1:
       return 217
    elif self.comboUlkeDeger==2:
       return 94
    elif self.comboUlkeDeger==3:
       return 153
    elif self.comboUlkeDeger==4:
       return 95
    elif self.comboUlkeDeger==5:
       return 4
    elif self.comboUlkeDeger==6:
       return 200
    else:
       return -1
    
    #209 Turkey
    #217 ABD
    #94 İran
    #153 Pakistan
    #95 Irak
    #4 Afganistan
    #200 Suriye


 def exportCSV(self):
    path = 'sonuclar'
    output_file = os.path.join(path,'Combined Book.csv')
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    self.predictions.toPandas().to_csv(export_file_path, sep=",", float_format='%.2f',index=False, line_terminator='\n',encoding='utf-8')
 
 def skorEkle(self):

         self.algoritma=self.combo.get()
         self.trainDataCount=self.trainingData.count()
         self.testDataCount=self.testData.count()
         self.dogrulukOrani=self.accuracy
         self.hataOrani=self.testError
         self.hesaplamaSuresi=self.tt
         self.egitilmeZamani=self.tt2
         self.f1Score=self.f1
         self.precisionSkor=self.wp
         self.recallScore=self.wr

         self.train_dogrulukOrani=self.train_accuracy
         self.train_hataOrani=self.train_Error
         self.train_hesaplamaSuresi=self.te
         self.train_egitilmeZamani=self.te2
         self.train_f1Score=self.train_f1
         self.train_precisionSkor=self.train_wp
         self.train_recallScore=self.train_wr

         self.tarihbug = str(datetime.now().strftime("%d.%m.%y_%H_%M"))
         temp1 = open("sonuclar.txt", "a")
         temp1.write("Algoritma:" +self.algoritma +" " +"Eğitim Data Sayısı: "  +str(self.trainDataCount) +" " +"Test Data Sayısı: " +str(self.testDataCount) +" " +"Dogruluk Orani: " +str(self.dogrulukOrani)
         +" " +"Hata Orani: " +str(self.hataOrani)  +" " +"Hesaplama Süresi: " +str(self.hesaplamaSuresi) +" sn "  +" " +"Egitilme zamani: " +str(self.egitilmeZamani) +" sn "   +" " +"F1 Skoru: " +str(self.f1Score)
         +" " +"Precision Skor: " +str(self.precisionSkor) +" " +"Recall Score: " +str(self.recallScore))
         temp1.write("\n")              
         messagebox.showinfo("Bilgi","%s algoritmasi listeye eklendi"%self.algoritma)
         path = "sonuclar"
         self.pathSave = path +'/' +self.algoritma+'_'+self.tarihbug +'.csv'
         
         with open(self.pathSave, mode='w') as csv_file:
             fieldnames = ['Algoritma', 'Data Sayısı', 'Dogruluk Orani', 'Hata Orani', 'Hesaplama Süresi', 'Egitilme zamani', 'F1 Skoru','Precision Skor', 'Recall Score']
             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
             writer.writeheader()

             writer.writerow({'Algoritma': ''+self.algoritma+' (Egitim) ', 'Data Sayısı': ''+str(self.trainDataCount), 'Dogruluk Orani': ''+str(self.train_dogrulukOrani),
                              'Hata Orani': ''+str(self.train_hataOrani), 'Hesaplama Süresi': ''+str(self.train_hesaplamaSuresi), 'Egitilme zamani': ''+str(self.train_egitilmeZamani), 'F1 Skoru': ''+str(self.train_f1Score),
                              'Precision Skor': ''+str(self.train_precisionSkor), 'Recall Score': ''+str(self.train_recallScore)})

             
             writer.writerow({'Algoritma': ''+self.algoritma+'(Test) ', 'Data Sayısı': ''+str(self.testDataCount), 'Dogruluk Orani': ''+str(self.dogrulukOrani),
                              'Hata Orani': ''+str(self.hataOrani), 'Hesaplama Süresi': ''+str(self.hesaplamaSuresi), 'Egitilme zamani': ''+str(self.egitilmeZamani), 'F1 Skoru': ''+str(self.f1Score),
                              'Precision Skor': ''+str(self.precisionSkor), 'Recall Score': ''+str(self.recallScore)})
             #writer.write("\n")


             messagebox.showinfo("Bilgi","%s algoritmasi CSV olarak eklendi"%self.algoritma)    




 def secVeGoster(self,event=None):
      self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
      print (self.filename)
      self.pathSave = self.filename
      self.csvView()

 def csvView(self):

   import tkinter
   import csv
   root = Tk()
   root.title("Sonuç Görüntüleme")
   path = "sonuclar"
   # open file
   with open(self.pathSave, mode='r') as file:
      reader = csv.reader(file)

      # r and c tell us where to grid the labels
      r = 0
      for col in reader:
         c = 0
         for row in col:
            # i've added some styling
            label = Label(root, width = 20, height = 3, \
                                  text = row, relief = tkinter.RIDGE)
            label.grid(row = r, column = c)
            c += 1
         r += 1

   root.mainloop()


 def listele(self):
        ListelePencere=Tk()
        ListelePencere.geometry("600x400")
        ListelePencere.title("Kişi Listeleme")
        ListelePencere.configure(background="red")
    
        liste=Text(ListelePencere,width="200",height="400",fg="white",bg="red",font="helvetica 12")
        liste.grid(row=0,column=0)
    
     
        satir_sayisi=0
        temp1 = open("sonuclar.txt", "r")
        readfile = temp1.read()
        liste.insert(END,readfile)

        
 def Hakkinda(self):
        HakkindaPencere=Tk()
        HakkindaPencere.geometry("700x50")
        HakkindaPencere.title("Barış KARABAY Fırat Üniversitesi Yazılım Mühendisliği Tez Projesi V2")
        HakkindaPencere.configure(background="red")

        self.baris=Label(HakkindaPencere,text="Bu Program Barış Karabay Tarafından Yapılmıştır \n Hiçbir Şekilde Paylaşılamaz ve Değiştirilemez. ",fg="black",bg="white")
        self.baris.grid(row=0,column=0)

 def secim(self,event=None):
      self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
      #ment = self.filename
      print (self.filename)
      #self.egitimTxt.set(self.filename)
      #self.['text']=self.filename
      self.nameText.set(self.filename)
      
 def secilenRadio1(self):
      print(self.selected1.get())
      if self.selected1.get()==1:
         #showinfo("Uyarı","birinci")
         self.combo.grid_remove()
      else:
         #self.combo.grid()
         self.combo.grid(column=3, row=2)

 def secilenRadio2(self):
      print(self.selected2.get())
      if self.selected2.get()==3:
         #showinfo("Uyarı","birinci")
         self.comboulke.grid_remove()
      else:
         #self.comboulke.grid()
         self.comboulke.grid(column=3, row=3)
                 
 def get_dummy(self):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col
    df = self.spark_df
    categoricalCols = self.catcols
    continuousCols = self.num_cols
    labelCol = self.labelCol
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in categoricalCols ]
    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    data = data.withColumn('label',col(labelCol))
    data.show(5,False)
    return data.select('features','label')


 def secilenDosya(self):
      print(self.egitimTxt.get())
      self.dosya=self.egitimTxt.get()
      self.dataSayisicntr = self.dataSayisiTxt.get()
      if self.dosya==" " or self.dosya=='' or self.dataSayisicntr=='':
         messagebox.showinfo("Uyarı","Boş Olamaz")
      else:
         print(self.comboulke.current(), self.comboulke.get())
         #self.progress.start()
         messagebox.showinfo("Uyarı","Yükleme Başlatıldı")
         #self.progress.config(mode='indeterminate')
         self.dosya = str(self.dosya)
         self.dataSayisi = int(self.dataSayisiTxt.get())
         print(self.dosya)
         mySchema = StructType([ StructField("country", IntegerType(), True)\
                       ,StructField("region", IntegerType(), True)\
                       ,StructField("attacktype1", IntegerType(), True)\
                       ,StructField("targtype1_txt", StringType(), True)\
                       ,StructField("gname", StringType(), True)\
                       ,StructField("weaptype1", IntegerType(), True)])
         
         
         #egitim=pd.read_csv("D:/globalterrorismdb2.csv", usecols=[7, 9, 26, 27, 28, 35, 36, 40, 58, 68, 81], encoding='ISO-8859-1',low_memory=False)
         self.egitim=pd.read_csv(self.dosya, usecols=[7, 9, 28, 35, 58, 81], encoding='ISO-8859-1',low_memory=False,nrows=self.dataSayisi)
         #209 Turkey
         #217 ABD
         #94 İran
         #153 Pakistan
         #95 Irak
         #4 Afganistan
         #200 Suriye
         if self.comboulke.get() != '' or self.comboulke.get() != "":
            self.egitim = self.egitim[(self.egitim.country == self.returnUlkeInt())]
            messagebox.showinfo("Bilgi","%s ülkesi için eğitim ve test veri seti oluşturulacak"%self.comboulke.get()) 
         
         print("Girilen  Sayi dogru")
         print("Toplam  Sayisi")
         print (self.egitim.count())
         self.sqlContext = SQLContext(self.sc)
         self.spark_df = self.sqlContext.createDataFrame(self.egitim, schema=mySchema)
         self.YukleBtn.grid_remove()
         #self.progress.stop()
         messagebox.showinfo("Başarılıı","Yükleme Tamamlandı")

 def DonusumuBaslat(self):
         sp_df = self.spark_df
         messagebox.showinfo("Uyarı","Dönüşüm Başladı")
         self.data_f = self.get_dummy()
         self.data_f.show(25,False)
         self.labelIndexer = StringIndexer(inputCol='label',outputCol='indexedLabel').fit(self.data_f)
         self.labelIndexer.transform(self.data_f).show(25,False)
         self.featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures",maxCategories=4).fit(self.data_f)
         self.featureIndexer.transform(self.data_f).show(25,False)
         self.labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=self.labelIndexer.labels)
         if self.testTxt.get()=='':
             messagebox.showinfo("Hata","Lütfen Test oranını girin")
         else:
            deger = self.testTxt.get()
            testPoint=float(deger)/100
            (self.trainingData, self.testData) = self.data_f.randomSplit([1.0-testPoint, testPoint], seed = 100)
            messagebox.showinfo("Başarılı","Oran Hesaplandı")
            self.DonusumBtn.grid_remove()

 def createFolder(self):
    try:
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
    except OSError:
        print ('Error: Creating directory. ' +  self.directory)

 def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

 def modeliEgit(self):
         print(self.combo.current(), self.combo.get())
         messagebox.showinfo("Bilgi","%s algoritması için model oluşturulacak"%self.combo.get()) 
         if self.combo.current()==0:
            self.LogicticRegressionClassifier()
         elif self.combo.current()==1: 
            self.NaiveBayesClassifier()
         elif self.combo.current()==2:
            self.RandomForestClassifier()
         elif self.combo.current()==3:
            self.DecisionTreeClassifier()
         elif self.combo.current()==4:
            self.SVMclassifier()
         elif self.combo.current()==5:
            self.KNNclassifier()
            
 def printMetrics(predictions_and_labels):
   metrics = MulticlassMetrics(predictions_and_labels)
   print('Precision of True ', metrics.precision(1))
   print('Precision of False', metrics.precision(0))
   print('Recall of True    ', metrics.recall(1))
   print('Recall of False   ', metrics.recall(0))
   print('F-1 Score         ', metrics.fMeasure())
   print('Confusion Matrix\n', metrics.confusionMatrix().toArray())
    
 def getPredictionsLabels(model, test_data):
   predictions = model.predict(test_data.map(lambda r: r.features))
   return predictions.zip(test_data.map(lambda r: r.label))
         
 def LogicticRegressionClassifier(self):
   self.t0 = time()
   print("********************************************************************************************************************************************")
   print("Logistic Regression")
   logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel',maxIter=20, regParam=0.3, elasticNetParam=0)
   pipeline = Pipeline(stages=[self.labelIndexer, self.featureIndexer, logr, self.labelConverter])
   model = pipeline.fit(self.trainingData)
   self.tm = time() - self.t0
   print ("Modeli egitme zamani {} saniye ".format(self.tm))
   self.t0 = time()
   self.predictions = model.transform(self.testData)
   self.tt = time() - self.t0
   print ("Test verisini siniflandirma zamani {} saniye ".format(self.tt))

   self.t0 = time()
   predictions_train = model.transform(self.trainingData)
   self.te = time() - self.t0
   print ("Egitim verisini siniflandirma zamani {} saniye ".format(self.te))
   
   self.predictions.select("features", "label", "predictedLabel", "probability").show(5)
   evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
   
   self.t0 = time()
   self.accuracy = evaluator.evaluate(self.predictions)
   self.tt2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, self.accuracy))
   
   self.t0 = time()
   self.train_accuracy = evaluator.evaluate(predictions_train)
   self.te2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Egitim Verisinin dogrulanmasi {} saniye ".format(self.te2, self.train_accuracy))
   
   print("Test Dogruluk = %g" % (self.accuracy))
   self.testError = (1.0 - self.accuracy)
   print("Test Test Error = %g" % (1.0 - self.accuracy))

   print("Egitim Dogruluk = %g" % (self.train_accuracy))
   self.train_Error = (1.0 - self.train_accuracy)
   print("Egitim Error = %g" % (1.0 - self.train_accuracy))

   rfModel = model.stages[2]
   evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
   self.f1 = evaluatorf1.evaluate(self.predictions)
   self.train_f1 = evaluatorf1.evaluate(predictions_train)
   print("test f1 = %g" % self.f1)
   print("egitim f1 = %g" % self.train_f1)
 
   evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
   self.wp = evaluatorwp.evaluate(self.predictions)
   self.train_wp = evaluatorwp.evaluate(predictions_train)
   print("test weightedPrecision = %g" % self.wp)
   print("egitim weightedPrecision = %g" % self.train_wp)
 
   evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
   self.wr = evaluatorwr.evaluate(self.predictions)
   self.train_wr = evaluatorwr.evaluate(predictions_train)
   print("test weightedRecall = %g" % self.wr)
   print("egitim weightedRecall = %g" % self.train_wr)

   rfModel = model.stages[2]
   #print (rfModel._call_java('toDebugString'))
   rfModel = model.stages[2]
   #model.save("model2345678909")
   messagebox.showinfo("Başarılı","Model Eğitildi")
   self.skorEkle()
   self.ModelBtn.grid_remove()
   self.SonucBtn.grid(row=7,column=2)
   self.ExportCsvBtn.grid(row=8,column=2)
   
   #self.predictions.printSchema()
   #paramGrid = (ParamGridBuilder()
   # .addGrid(logr.regParam, [0.01, 0.1, 0.5]) \
   #  .addGrid(logr.maxIter, [10, 20, 50]) \
   #  .addGrid(logr.elasticNetParam, [0.0, 0.8]) \
   # .build())
   
   #crossval = CrossValidator(estimator=pipeline,
   #                       estimatorParamMaps=paramGrid,
   #                       evaluator=evaluator,
   #                       numFolds=3)
   #
   #model = crossval.fit(self.trainingData)
   #predictions = model.transform(self.testData)
   #accuracy = evaluator.evaluate(predictions)
   #print("Dogruluk = %g" % (accuracy))

 def DecisionTreeClassifier(self):
   self.t0 = time()
   print("********************************************************************************************************************************************")
   print("Decision Tree Classifier")
   dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",impurity="gini",maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                         cacheNodeIds=False, checkpointInterval=10)
   pipeline = Pipeline(stages=[self.labelIndexer, self.featureIndexer, dt, self.labelConverter])
   model = pipeline.fit(self.trainingData)
   self.tm = time() - self.t0
   print ("Modeli egitme zamani {} saniye ".format(self.tm))

   self.t0 = time()
   self.predictions = model.transform(self.testData)
   self.tt = time() - self.t0
   print ("Test verisini siniflandirma zamani {} saniye ".format(self.tt))

   self.t0 = time()
   predictions_train = model.transform(self.trainingData)
   self.te = time() - self.t0
   print ("Egitim verisini siniflandirma zamani {} saniye ".format(self.te))
   
   self.predictions.select("features", "label", "predictedLabel", "probability").show(5)
   evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
   
   self.t0 = time()
   self.accuracy = evaluator.evaluate(self.predictions)
   self.tt2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, self.accuracy))
   
   self.t0 = time()
   self.train_accuracy = evaluator.evaluate(predictions_train)
   self.te2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Egitim Verisinin dogrulanmasi {} saniye ".format(self.te2, self.train_accuracy))
   
   print("Test Dogruluk = %g" % (self.accuracy))
   self.testError = (1.0 - self.accuracy)
   print("Test Test Error = %g" % (1.0 - self.accuracy))

   print("Egitim Dogruluk = %g" % (self.train_accuracy))
   self.train_Error = (1.0 - self.train_accuracy)
   print("Egitim Error = %g" % (1.0 - self.train_accuracy))

   rfModel = model.stages[2]
   evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
   self.f1 = evaluatorf1.evaluate(self.predictions)
   self.train_f1 = evaluatorf1.evaluate(predictions_train)
   print("test f1 = %g" % self.f1)
   print("egitim f1 = %g" % self.train_f1)
 
   evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
   self.wp = evaluatorwp.evaluate(self.predictions)
   self.train_wp = evaluatorwp.evaluate(predictions_train)
   print("test weightedPrecision = %g" % self.wp)
   print("egitim weightedPrecision = %g" % self.train_wp)
 
   evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
   self.wr = evaluatorwr.evaluate(self.predictions)
   self.train_wr = evaluatorwr.evaluate(predictions_train)
   print("test weightedRecall = %g" % self.wr)
   print("egitim weightedRecall = %g" % self.train_wr)

   rfModel = model.stages[2]
   #print (rfModel._call_java('toDebugString'))
   messagebox.showinfo("Başarılı","Model Eğitildi")
   self.skorEkle()
   self.ModelBtn.grid_remove()
   self.SonucBtn.grid(row=7,column=2)
   self.ExportCsvBtn.grid(row=8,column=2)
   
 def NaiveBayesClassifier(self):
   print("********************************************************************************************************************************************")
   self.t0 = time()
   print("Bayes")
   nb = NaiveBayes(featuresCol='indexedFeatures', labelCol='indexedLabel', smoothing=1.0, modelType="multinomial")
   pipeline = Pipeline(stages=[self.labelIndexer, self.featureIndexer, nb, self.labelConverter])
   model = pipeline.fit(self.trainingData)
   self.tm = time() - self.t0
   print ("Modeli egitme zamani {} saniye ".format(self.tm))

   self.t0 = time()
   self.predictions = model.transform(self.testData)
   self.tt = time() - self.t0
   print ("Test verisini siniflandirma zamani {} saniye ".format(self.tt))

   self.t0 = time()
   predictions_train = model.transform(self.trainingData)
   self.te = time() - self.t0
   print ("Egitim verisini siniflandirma zamani {} saniye ".format(self.te))
   
   self.predictions.select("features", "label", "predictedLabel", "probability").show(5)
   evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
   
   self.t0 = time()
   self.accuracy = evaluator.evaluate(self.predictions)
   self.tt2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, self.accuracy))
   
   self.t0 = time()
   self.train_accuracy = evaluator.evaluate(predictions_train)
   self.te2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Egitim Verisinin dogrulanmasi {} saniye ".format(self.te2, self.train_accuracy))
   
   print("Test Dogruluk = %g" % (self.accuracy))
   self.testError = (1.0 - self.accuracy)
   print("Test Test Error = %g" % (1.0 - self.accuracy))

   print("Egitim Dogruluk = %g" % (self.train_accuracy))
   self.train_Error = (1.0 - self.train_accuracy)
   print("Egitim Error = %g" % (1.0 - self.train_accuracy))

   rfModel = model.stages[2]
   evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
   self.f1 = evaluatorf1.evaluate(self.predictions)
   self.train_f1 = evaluatorf1.evaluate(predictions_train)
   print("test f1 = %g" % self.f1)
   print("egitim f1 = %g" % self.train_f1)
 
   evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
   self.wp = evaluatorwp.evaluate(self.predictions)
   self.train_wp = evaluatorwp.evaluate(predictions_train)
   print("test weightedPrecision = %g" % self.wp)
   print("egitim weightedPrecision = %g" % self.train_wp)
 
   evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
   self.wr = evaluatorwr.evaluate(self.predictions)
   self.train_wr = evaluatorwr.evaluate(predictions_train)
   print("test weightedRecall = %g" % self.wr)
   print("egitim weightedRecall = %g" % self.train_wr)

   #print (rfModel._call_java('toDebugString'))
   messagebox.showinfo("Başarılı","Model Eğitildi")
   self.skorEkle()
   self.ModelBtn.grid_remove()
   self.SonucBtn.grid(row=7,column=2)
   self.ExportCsvBtn.grid(row=8,column=2)
 def RandomForestClassifier(self):
   print("********************************************************************************************************************************************")
   print("Random Forest")
   self.t0 = time()
   rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees = 100, maxDepth = 4, maxBins = 32,impurity="entropy")
   pipeline = Pipeline(stages=[self.labelIndexer, self.featureIndexer, rf, self.labelConverter])
   model = pipeline.fit(self.trainingData)
   self.tm = time() - self.t0
   print ("Modeli egitme zamani {} saniye ".format(self.tm))

   self.t0 = time()
   self.predictions = model.transform(self.testData)
   self.tt = time() - self.t0
   print ("Test verisini siniflandirma zamani {} saniye ".format(self.tt))

   self.t0 = time()
   predictions_train = model.transform(self.trainingData)
   self.te = time() - self.t0
   print ("Egitim verisini siniflandirma zamani {} saniye ".format(self.te))
   
   self.predictions.select("features", "label", "predictedLabel", "probability").show(5)
   evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
   
   self.t0 = time()
   self.accuracy = evaluator.evaluate(self.predictions)
   self.tt2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, self.accuracy))
   
   self.t0 = time()
   self.train_accuracy = evaluator.evaluate(predictions_train)
   self.te2 = time() -self.t0
   print ("Tahmini yapilis zamani {} saniye . Egitim Verisinin dogrulanmasi {} saniye ".format(self.te2, self.train_accuracy))
   
   print("Test Dogruluk = %g" % (self.accuracy))
   self.testError = (1.0 - self.accuracy)
   print("Test Test Error = %g" % (1.0 - self.accuracy))

   print("Egitim Dogruluk = %g" % (self.train_accuracy))
   self.train_Error = (1.0 - self.train_accuracy)
   print("Egitim Error = %g" % (1.0 - self.train_accuracy))

   rfModel = model.stages[2]
   evaluatorf1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
   self.f1 = evaluatorf1.evaluate(self.predictions)
   self.train_f1 = evaluatorf1.evaluate(predictions_train)
   print("test f1 = %g" % self.f1)
   print("egitim f1 = %g" % self.train_f1)
 
   evaluatorwp = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
   self.wp = evaluatorwp.evaluate(self.predictions)
   self.train_wp = evaluatorwp.evaluate(predictions_train)
   print("test weightedPrecision = %g" % self.wp)
   print("egitim weightedPrecision = %g" % self.train_wp)
 
   evaluatorwr = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
   self.wr = evaluatorwr.evaluate(self.predictions)
   self.train_wr = evaluatorwr.evaluate(predictions_train)
   print("test weightedRecall = %g" % self.wr)
   print("egitim weightedRecall = %g" % self.train_wr)

   rfModel = model.stages[2]
   #print (rfModel._call_java('toDebugString'))
   messagebox.showinfo("Başarılı","Model Eğitildi")
   self.skorEkle()
   self.ModelBtn.grid_remove()
   self.SonucBtn.grid(row=7,column=2)
   self.ExportCsvBtn.grid(row=8,column=2)
   
   svm = LinearSVC(maxIter=5, regParam=0.01)
   LSVC = LinearSVC()
   ovr = OneVsRest(classifier=LSVC)
   paramGrid = ParamGridBuilder().addGrid(LSVC.maxIter, [10, 100]).addGrid(LSVC.regParam,[0.001, 0.01, 1.0,10.0]).build()
   crossval = CrossValidator(estimator=ovr,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=MulticlassClassificationEvaluator(metricName="f1"),
                                  numFolds=2)
   Train_sparkframe = self.trainingData.select("features", "label")
   cvModel = crossval.fit(Train_sparkframe)
   bestModel = cvModel.bestModel

   

 def SVMclassifier(self):
   print("********************************************************************************************************************************************")
   self.t0 = time()
   print("SVM")
   df = self.egitim
   df['gname_id'] = df['gname'].factorize()[0]
   df['weaptype1_id'] = df['weaptype1'].factorize()[0]
   df['targtype1_txt_id'] = df['targtype1_txt'].factorize()[0]
   df['targsubtype1_id'] = df['targsubtype1'].factorize()[0]
   X = df.iloc[:, [0,1,2,8,9,10]].values
   y = df.iloc[:, 7].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
   scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
   X_train = scaling.transform(X_train)
   X_test = scaling.transform(X_test)
   classifier = SVC(kernel='linear',cache_size=7000, random_state = 0)
   classifier.fit(X_train, y_train)
   self.tt = time() - self.t0
   print ("Egitim verisini siniflandirma zamani {} saniye ".format(self.tt))
   self.t0 = time()
   y_pred = classifier.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   self.tt2 = time() -self.t0
   print(accuracy)
   print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, accuracy))

 def KNNclassifier(self):
   print("********************************************************************************************************************************************")
   from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.svm import SVC
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.naive_bayes import GaussianNB
   from sklearn.metrics import classification_report
   from sklearn.metrics import confusion_matrix
   from sklearn.preprocessing import LabelEncoder

   
   print("KNN")
   df = self.egitim

   df['gname_id'] = df['gname'].factorize()[0]
   df['weaptype1_id'] = df['weaptype1'].factorize()[0]
   df['targtype1_txt_id'] = df['targtype1_txt'].factorize()[0]
   print("Toplam  Sayisi")
   #print (df.count())

   X = df.iloc[:, [0,1,2,7,8]].values
   y = df.iloc[:, 6].values
   #print(df.iloc[:, 6])
   #print(df.columns)
   #print(X)
   #print(y)
   #print(df['gname_id'])
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
   scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
   X_train = scaling.transform(X_train)
   X_test = scaling.transform(X_test)
   classifier = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p = 2)
   self.t0 = time()
   classifier.fit(X_train, y_train)
   self.tt = time() - self.t0
   print ("Veri kumesini egitim zamani {} saniye ".format(self.tt))
   self.t0 = time()
   y_pred = classifier.predict(X_test)
   self.tt = time() - self.t0
   print ("test verisini siniflandirma zamani {} saniye ".format(self.tt))
   self.t0 = time()
   x_pred = classifier.predict(X_train)
   self.tt = time() - self.t0
   print ("egitim verisini siniflandirma zamani {} saniye ".format(self.tt))

   accuracy = accuracy_score(y_test, y_pred)
   accuracy_egitim = accuracy_score(y_train, x_pred)
   self.tt2 = time() -self.t0

   print ('Test Accuracy:', accuracy)
   print ('Egitim Accuracy:', accuracy_egitim)
   #print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, accuracy))
   #print 'Accuracy:', accuracy_score(y_test, y_pred)
   print ('Test F1 score:', f1_score(y_test, y_pred,average='weighted'))
   print ('Test Recall:', recall_score(y_test, y_pred,
                                 average='weighted'))
   print ('Test Precision:', precision_score(y_test, y_pred,
                                       average='weighted'))

   print ('egitim F1 score:', f1_score(y_train, x_pred,average='weighted'))
   print ('egitim Recall:', recall_score(y_train, x_pred,
                                 average='weighted'))
   print ('egitim Precision:', precision_score(y_train, x_pred,
                                       average='weighted'))

   #print '\n clasification report:\n', classification_report(y_test, y_pred)
   #print '\n confussion matrix:\n',confusion_matrix(y_test, y_pred)

   print("********************************************************************************************************************************************")
   #sys.exit('bittttiiiii')
   
   ##   print("********************************************************************************************************************************************")
   ##   print("SVM Classifier")
   ##   classifier = SVC(kernel='linear',cache_size=7000, random_state = 0)
   ##   self.t0 = time()
   ##   classifier.fit(X_train, y_train)
   ##   self.tt = time() - self.t0
   ##   print ("Veri kumesini egitim zamani {} saniye ".format(self.tt))
   ##   self.t0 = time()
   ##   y_pred = classifier.predict(X_test)
   ##   self.tt = time() - self.t0
   ##   print ("test verisini siniflandirma zamani {} saniye ".format(self.tt))
   ##   self.t0 = time()
   ##   x_pred = classifier.predict(X_train)
   ##   self.tt = time() - self.t0
   ##   print ("egitim verisini siniflandirma zamani {} saniye ".format(self.tt))
   ##
   ##   accuracy = accuracy_score(y_test, y_pred)
   ##   accuracy_egitim = accuracy_score(y_train, x_pred)
   ##   self.tt2 = time() -self.t0
   ##
   ##   print ('Test Accuracy:', accuracy)
   ##   print ('Egitim Accuracy:', accuracy_egitim)
   ##   #print ("Tahmini yapilis zamani {} saniye . Testin dogrulanmasi {} saniye ".format(self.tt2, accuracy))
   ##   #print 'Accuracy:', accuracy_score(y_test, y_pred)
   ##   print ('Test F1 score:', f1_score(y_test, y_pred,average='weighted'))
   ##   print ('Test Recall:', recall_score(y_test, y_pred,
   ##                                 average='weighted'))
   ##   print ('Test Precision:', precision_score(y_test, y_pred,
   ##                                       average='weighted'))
   ##
   ##   print ('egitim F1 score:', f1_score(y_train, x_pred,average='weighted'))
   ##   print ('egitim Recall:', recall_score(y_train, x_pred,
   ##                                 average='weighted'))
   ##   print ('egitim Precision:', precision_score(y_train, x_pred,
   ##                                       average='weighted'))


adresDefteri()
