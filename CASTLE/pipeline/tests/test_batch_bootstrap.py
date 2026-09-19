import json
import os
import pytest
from bootstrap_batch import main


def test_adc_materialized_privately_and_secret_removed_before_exec(tmp_path,monkeypatch):
    target=tmp_path/'private'/'adc.json'
    fake={'type':'authorized_user','client_id':'offline-fixture','refresh_token':'offline-fixture'}
    monkeypatch.setenv('GOOGLE_ADC_JSON',json.dumps(fake))
    monkeypatch.setenv('CASTLE_ADC_SECRET_NAME','GOOGLE_ADC_JSON')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',str(target))
    monkeypatch.setenv('GOOGLE_API_KEY','offline-key')
    monkeypatch.setenv('GEMINI_API_KEY','offline-key')
    calls=[]
    def execute(executable,args):
        assert 'GOOGLE_ADC_JSON' not in os.environ
        assert 'GOOGLE_API_KEY' not in os.environ and 'GEMINI_API_KEY' not in os.environ
        assert json.loads(target.read_text())==fake
        calls.append((executable,args))
    monkeypatch.setattr(os,'execvp',execute)
    main(['python','worker.py'])
    assert calls==[('python',['python','worker.py'])]
    if os.name!='nt':assert target.stat().st_mode&0o777==0o600


def test_adc_does_not_replace_existing_file(tmp_path,monkeypatch):
    target=tmp_path/'adc.json';target.write_text('existing')
    monkeypatch.setenv('GOOGLE_ADC_JSON','{"type":"service_account"}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS',str(target))
    monkeypatch.delenv('CASTLE_ADC_SECRET_NAME',raising=False)
    monkeypatch.setattr(os,'execvp',lambda *args:pytest.fail('must not execute'))
    with pytest.raises(FileExistsError):main(['python','worker.py'])
    assert target.read_text()=='existing'
