"""Materialize one HF secret as private ADC, then replace this process."""
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    command=sys.argv[1:] if argv is None else argv
    if not command:
        raise ValueError('Missing worker command')
    key=os.environ.pop('CASTLE_ADC_SECRET_NAME','GOOGLE_ADC_JSON')
    value=os.environ.pop(key,None)
    if not value:
        raise ValueError('Google ADC secret is missing')
    data=json.loads(value)
    if not isinstance(data,dict) or data.get('type') not in {'service_account','authorized_user','external_account','impersonated_service_account'}:
        raise ValueError('Expected Google ADC JSON, not an API key')
    target=Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:
        json.dump(data,handle)
    # Ambient online keys must not select a different authentication mechanism.
    os.environ.pop('GOOGLE_API_KEY',None);os.environ.pop('GEMINI_API_KEY',None)
    os.execvp(command[0],command)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('Batch credential bootstrap failed: '+type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
