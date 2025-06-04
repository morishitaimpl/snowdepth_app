# snowdepth_app
積雪深予測アプリケーション
このリポジトリのソースコードはDevinで作成しました。

必要なライブラリは次のようにコマンド入力してください。
pip install -r library_install.txt

train_model.pyをまずは実行してください。
.pklファイルが作成されればOKです。
なお、train_modelpyで学習用のcsvファイル置き場を指定しているため、修正するかもしくはソースコードに倣う必要があります。

次にstreamlit_app.pyを実行します。
streamlitライブラリで実装しているため、pythonコマンドではなく、streamlit run streamlit_app.pyと入力してください。

選択した都市に応じてモデルも変更できるよう、今後は都市ごとにページ遷移できるように分別する予定です。
