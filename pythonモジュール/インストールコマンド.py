cd C:\Users\t0915526\Downloads
#whlファイル
python -m pip install --no-deps requests-2.28.2-py3-none-any.whl

pip install --no-deps urllib-3-0.1.tar.gz --use-deprecated=legacy-resolver



#pip cp確認コマンド
from pip._internal.utils.compatibility_tags import get_supported
get_supported()


#バージョンが一緒なのにエラーの場合
#numpy?1.11.2+mkl?cp35?cp35m?win_amd64.whl　というファイル名だった場合，
#numpy?1.11.2+mkl?cp35?none?win_amd64.whl に変更する