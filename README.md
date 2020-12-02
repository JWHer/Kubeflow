# Kubeflow 파이프라인 사용해보기
이 프로젝트는 [SimpleTransformer를 사용한 문맥분석 프로젝트](https://github.com/JWHer)의 파이프라인화 과정을 다룬다. Deep Learning, Google Cloud, Kubernetes에 대한 이해가 있어야 한다.

## 1. Pipeline이 왜 필요한데?
*딥러닝은 수학 이론과 학습 모델만 알면 되는거 아니야?*

2016년 알파고 등장 후 대중에 떠오르는 딥러닝에 대해 이론과 기초적인 모델에 대해 공부했다. 개인 프로젝트와 연구를 하며 이런저런 시행착오를 겪으면서 CNN모델과 BERT모델을 만들며 모델 생성에는 자신감이 생겼다.

<p align="center">
<image src="https://www.sciencetimes.co.kr/wp-content/uploads/2020/03/thumb_400.jpg" width="50%">
</p>
<p align="center"><i>딥러닝은 좀 알겠는데...</i></p>

하지만 위 프로젝트들은 생성된 모델은 아무리 성능이 좋아도 치명적인 단점들이 있다. 데이터 셋이 변하면 작동하지 않는 것, 모델 확장이 어렵다는 것. 특히 실시간으로 변하는 데이터를 다루면 매번 모델을 재학습시켜 교체하는 것은 불가능 할 것이다.*(ex youtube...)* 실무에서 요구하는 변경과 통합에 유연하게 대처 가능한 pipeline이 필요한 것이다.



## 2. Kubeflow
*Kubeflow 참 직관적인 이름이다.*

Kubeflow라는 이름과 같이 머신러닝 workflow를 kubernetes에서 쉽게 배포할 수 있게 만드는 것이다. [Kubeflow 공식 documentation](https://www.kubeflow.org/docs/about/kubeflow/)에서 3가지 목표를 든다.

 - 쉽고, 반복가능하고, 이식하기 쉬운 배포 기반제공.
 - 느슨한 결합으로 소규모 서비스 배포와 관리
 - 필요에 따른 확장

(오역이 있을 수 있습니다. 원문을 봐주세요)
궁극적으로, Kubernetes를 사용하고 있는 곳에서 ML스택을 제공하는 것을 목표로 다양한 어플리케이션을 제공한다. 하지만 본 프로젝트에선 pipeline을 사용한 머신러닝 시스템을 만드는 것이므로 Kubeflow pipelines만 다룬다.

## 3. 직접 해보기
*천리길도 한걸음부터*

PC에 설치할 수도 있겠지만 성능 문제로 [Google Cloud 문서](https://cloud.google.com/ai-platform/pipelines/docs/getting-started)를 따라간다.

1. Google Cloud 사용
Google Cloud에 Kubeflow가 동작할 환경을 만들어주자. 첫 사용자에게 몇개월간 300$의 크레딧을 지원해줘 무료로 사용 가능하다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치1.png" height="300"></p>
<p align="center"><i> Trail 기간과 credit을 다 써서 결제 해야한다...</i></p>

2. AI Platform 파이프라인 인스턴스 설정
<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치2.png" height="300"></p>
Google Cloud Console에서 AI Platform 파이프라인을 연다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치3.png" height="300"></p>
사용할 Google Cloud 프로젝트를 선택한 다음 열기를 클릭한다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치4.png" height="300"></p>
AI Platform Pipelines 툴바에서 새 인스턴스를 클릭한다. Google Cloud Marketplace에서 Kuberflow Piplelines가 열린다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치5.png" height="300"></p>
구성을 클릭한다. 배포 구성 양식이 열린다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치6.png" height="300"></p>
<p align="center"><i>올 초에 드디어 한국에도 Cloud 서버가 생겼다</i></p>
클러스터 영역을 설정하고, 다음 Cloud API에 대한  엑세스 허용을 선택한다. 이후 클러스터 만들기를 클릭한다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치7.png" height="300"></p>
<center><i>이름은 원하는대로 지었다. 잘 기억해두자.</i></center>
클러스터를 만든 후 네임스페이스(default)와 앱 인스턴스 이름을 제공한다. 이후 배포를 누른다.

<p align="center"><i>이쯤에서 쉬어가는 게 좋을 것이라고 생각한다...</i></p>

3. Cloud Storage에 작업 bucket 생성 및 데이터 업로드
<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/설치8.png" height="300"></p>
다시 AI Platform Pipelines로 돌아와 파이프라인 대시보드 열기를 클릭한다. Kubeflow Pipelines 대시보드가 열리고 시작하기 페이지가 표시된다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/저장소1.png" height="300"></p>
AI Platform Pipelines를 설치하면 Google Cloud Storage 에 자동으로 버킷이 생성된다. 이름을 클릭한다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/저장소2.png" height="300"></p>
필요한 데이터를 업로드한다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북1.png" height="300"></p>
기존에 작성했던 노트북 코드를 옮길 것이다. 또한, Jupyter를 사용하면 다른 장점도 많다.(물론 단점도 있겠지만...)

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북2.png" height="300"></p
메모장 인스턴스를 생성해 준다. 무료 체험도 끝났고 돈이 없기때문에... 가장 저렴한 머신을 사용한다. 이전에 노트북을 사용했던 이유도 [colab](https://colab.research.google.com/) 환경에서 무료로 작업했기 때문이다. (아직 머신러닝 기초를 공부하는 단계면 강추한다.)

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북3.png" height="300"></p>
마저 continue를 눌러 완료하자.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북4.png" width="40%"></p>
<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북5.png" width="40%"></p>
이제 익숙한 노트북 환경이 보인다!

4. Kubeflow로 이전하기
[여기](https://medium.com/google-cloud-apac/gcp-ai-platform-%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%98%EB%8A%94-kubeflow-pipelines-%EA%B8%B0%EB%B0%98-ml-%ED%95%99%EC%8A%B5-%EB%B0%8F-%EB%B0%B0%ED%8F%AC-%EC%98%88%EC%A0%9C-part-2-3-22b597f8d127)를 따라간다.

모델은 전처리, 학습, 배포의 단계로 나눌 수 있다. 하지만 [이전 프로젝트](https://github.com/JWHer)의 데이터셋은 이미 처리되었기 때문에 전처리 단계는 생략한다. 전처리된 데이터를 Cloud Storage에서 다운받아 학습한다. 정확도가 더 높아진 경우 생성된 모델을 다시 Cloud Storage에 업로드하게 된다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북6.png" height="300"></p>
<p align="center"><i>코드는 리팩토링이 좀 필요할듯...</i></p>
코드는 똑같다. 단지 저장 위치 Cloud Storage가 되도록 수정해 주었다.

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/노트북7.png" height="300"></p>
Dokerfile을 생성해준다. pipeline.ipynb에서 실행이 잘 되는지 테스트 해 보았다.

클라우드 상에서 학습을 할 때 **패키지를 읽어** 수행하게 된다. 따라서 패키기를 만들기 위해 setup.py 생성, 압축, 업로드 작업이 필요하다.

    !rm -fr titanic_train.tar.gz  
    !tar zcvf titanic_train.tar.gz *  
    !gsutil cp titanic_train.tar.gz $AIPJOB_TRAINER_GCS_PATH
<p align="center"><i>열심히 따라해보자</i></p>

5.  Kubeflow Pipeline 구성 코드 작성
[여기](https://medium.com/google-cloud-apac/gcp-ai-platform-%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%98%EB%8A%94-kubeflow-pipelines-%EA%B8%B0%EB%B0%98-ml-%ED%95%99%EC%8A%B5-%EB%B0%8F-%EB%B0%B0%ED%8F%AC-%EC%98%88%EC%A0%9C-part-3-3-87ff52f8507a)를 따라간다

<details>
<summary>원본 소스</summary>
<div markdown="1">

    #titanic_kfp_pipeline.ipynb  
    #Copyright 2020 Google LLC.   
    #This software is provided as-is, without warranty or representation for any use or purpose.   
    #Your use of it is subject to your agreements with Google.  
    #Author: whjang@google.com#!pip3 install -U kfp  
    import kfp  
    import kfp.components as comp  
    from kfp import dsl  
    from kfp import compiler  
    from kfp.components import func_to_container_op  
    import time  
    import datetimePIPELINE_HOST = “55b5c3378a14c1c1-dot-us-west1.pipelines.googleusercontent.com”  
    WORK_BUCKET = “gs://aiplatformdemo-kubeflowpipelines-default”  
    EXPERIMENT_NAME = “Titanic Draft Experiment”# Function for determine deployment  
    @func_to_container_op  
    def check_and_deploy_op(ACC_CSV_GCS_URI) -> str:  
     import sys, subprocess  
     subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘pandas’])  
     subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘gcsfs’])  
     import pandas as pd  
     acc_df = pd.read_csv(ACC_CSV_GCS_URI)  
     return acc_df[“deploy”].item()@func_to_container_op  
    def finish_deploy_op(ACC_CSV_GCS_URI):  
     import sys, subprocess  
     subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘pandas’])  
     subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘gcsfs’])  
     import pandas as pd  
     acc_df = pd.read_csv(ACC_CSV_GCS_URI)  
     acc_df[“deploy”] = “done”  
     acc_df.to_csv(ACC_CSV_GCS_URI)  
     print(“Successfully new model was deployed”)@dsl.pipeline(  
     name=”titanic-kubeflow-pipeline-demo”,  
     description = “Titanic Kubeflow Pipelines demo embrassing AI Platform in Google Cloud”  
    )def titanic_pipeline(  
     PROJECT_ID,  
     WORK_BUCKET,  
     RAW_CSV_GCS_URI,  
     PREPROC_CSV_GCS_URI,  
     ACC_CSV_GCS_URI,  
     MODEL_PKL_GCS_URI,  
     MIN_ACC_PROGRESS,  
     STAGE_GCS_FOLDER,  
     TRAIN_ON_CLOUD,  
     AIPJOB_TRAINER_GCS_PATH,  
     AIPJOB_OUTPUT_GCS_PATH  
    ):  
     IMAGE_PREFIX = “whjang-titanic”  
     PREPROC_DIR = “preprocess”  
     TRAIN_DIR = “train”  
     MODEL_DIR = “model”  
       
     preprocess = dsl.ContainerOp(  
     name = “Preprocess raw data and generate new one”,  
     image = “gcr.io/” + str(PROJECT_ID) + “/” + IMAGE_PREFIX + “-” + PREPROC_DIR + “:latest”,  
     arguments = [  
     “--raw_csv_gcs_uri”, RAW_CSV_GCS_URI,  
     “--preproc_csv_gcs_uri”, PREPROC_CSV_GCS_URI  
     ]  
     ) train_args = [  
     “--preproc_csv_gcs_uri”, str(PREPROC_CSV_GCS_URI),  
     “--model_pkl_gcs_uri”, str(MODEL_PKL_GCS_URI),  
     “--acc_csv_gcs_uri”, str(ACC_CSV_GCS_URI),  
     “--min_acc_progress”, str(MIN_ACC_PROGRESS)  
     ]  
       
     with dsl.Condition(TRAIN_ON_CLOUD == False) as check_condition1:  
     train = dsl.ContainerOp(  
     name = “Train”,  
     image = “gcr.io/” + str(PROJECT_ID) + “/” + IMAGE_PREFIX + “-” + TRAIN_DIR + “:latest”,  
     arguments = train_args,  
     file_outputs={  
     “mlpipeline-metrics” : “/mlpipeline-metrics.json”  
     }  
     )  
       
     with dsl.Condition(TRAIN_ON_CLOUD == True) as check_condition2:  
     aip_job_train_op = comp.load_component_from_url(“https://raw.githubusercontent.com/kubeflow/pipelines/1.0.0/components/gcp/ml_engine/train/component.yaml”)  
     help(aip_job_train_op)  
     aip_train = aip_job_train_op(  
     project_id=PROJECT_ID,   
     python_module=”train.titanic_train”,   
     package_uris=json.dumps([str(AIPJOB_TRAINER_GCS_PATH)]),   
     region=”us-west1",   
     args=json.dumps(train_args),  
     job_dir=AIPJOB_OUTPUT_GCS_PATH,   
     python_version=”3.7",  
     runtime_version=”1.15", #cf. 2.1   
     master_image_uri=””,   
     worker_image_uri=””,   
     training_input=””,   
     job_id_prefix=””,   
     job_id=””,  
     wait_interval=5  
     )  
       
     check_deploy = check_and_deploy_op(ACC_CSV_GCS_URI)  
     with dsl.Condition(check_deploy.output == “pending”):  
     aip_model_deploy_op = comp.load_component_from_url(“https://raw.githubusercontent.com/kubeflow/pipelines/1.0.0/components/gcp/ml_engine/deploy/component.yaml”)  
     help(aip_model_deploy_op)  
     aip_model_deploy = aip_model_deploy_op(  
     model_uri=str(WORK_BUCKET) + “/” + MODEL_DIR,   
     project_id=PROJECT_ID,   
     model_id=””,   
     version_id=””,   
     runtime_version=”1.15", #cf. 2.1   
     python_version=”3.7",  
     version=””,   
     replace_existing_version=”False”,   
     set_default=”True”,   
     wait_interval=5  
     )  
     lastStep = finish_deploy_op(ACC_CSV_GCS_URI)  
       
     check_condition1.after(preprocess)  
     check_condition2.after(preprocess)  
     check_deploy.after(aip_train)  
     lastStep.after(aip_model_deploy)  
       
     train.execution_options.caching_strategy.max_cache_staleness = “P0D”  
     aip_train.execution_options.caching_strategy.max_cache_staleness = “P0D”  
     check_deploy.execution_options.caching_strategy.max_cache_staleness = “P0D”  
     aip_model_deploy.execution_options.caching_strategy.max_cache_staleness = “P0D”  
     lastStep.execution_options.caching_strategy.max_cache_staleness = “P0D”  
       
    args = {  
     “PROJECT_ID” : “aiplatformdemo”,  
     “WORK_BUCKET” : WORK_BUCKET,  
     “RAW_CSV_GCS_URI” : WORK_BUCKET + “/rawdata/train.csv”,  
     “PREPROC_CSV_GCS_URI” : WORK_BUCKET + “/preprocdata/processed_train.csv”,  
     “ACC_CSV_GCS_URI” : WORK_BUCKET + “/latestacc/accuracy.csv”,  
     “MODEL_PKL_GCS_URI” : WORK_BUCKET + “/model/model.pkl”,  
     “MIN_ACC_PROGRESS” : 0.000001,  
     “STAGE_GCS_FOLDER” : WORK_BUCKET + “/stage”,  
     “TRAIN_ON_CLOUD” : False,  
     “AIPJOB_TRAINER_GCS_PATH” : WORK_BUCKET + “/train/titanic_train.tar.gz”,  
     “AIPJOB_OUTPUT_GCS_PATH” : WORK_BUCKET + “/train/output/”  
    }client = kfp.Client(host=PIPELINE_HOST)  
    #pipeline_name = “titanic_pipelines.zip”  
    #compiler.Compiler().compile(titanic_pipeline, pipeline_name)  
    #try:  
    # pipeline = client.upload_pipeline(pipeline_package_path=pipeline_name, pipeline_name=pipeline_name)  
    # print(“uploaded:” + pipeline.id)  
    #except:  
    # print(“already exist”)client.create_run_from_pipeline_func(  
     titanic_pipeline,  
     arguments=args,  
     experiment_name=EXPERIMENT_NAME  
    )

</div>
</details>

6. Kubeflow cluster로 보기

<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/실험1.png" height="300"></p>
 
<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/실험2.png" height="300"></p>
 
<p align="center"><image src="https://raw.githubusercontent.com/JWHer/Kubeflow/main/image/실험3.png" height="300"></p>

## 참고 사이트
[1] https://medium.com/daangn/kubeflow-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8-%EC%9A%B4%EC%9A%A9%ED%95%98%EA%B8%B0-6c6d7bc98c30

[2] https://medium.com/google-cloud-apac/gcp-ai-platform-%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%98%EB%8A%94-kubeflow-pipelines-%EA%B8%B0%EB%B0%98-ml-%ED%95%99%EC%8A%B5-%EB%B0%8F-%EB%B0%B0%ED%8F%AC-%EC%98%88%EC%A0%9C-part-1-3-d49f1096d786
