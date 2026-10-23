import io
import tarfile
import pytest
from rosorin_vla.download import extract_model

def test_archive_traversal_rejected(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        member=tarfile.TarInfo('../escape');member.size=1;tar.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='UNSAFE_ARCHIVE_PATH'):
        extract_model(archive,tmp_path/'models',{'model_files':{}})
    assert not (tmp_path/'escape').exists()
