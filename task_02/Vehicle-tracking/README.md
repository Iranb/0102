```
pip install -r requirements.txt
```

    
## Config

`settings/`
<br></br>
`config.yml`  `deepsort.yml`  `db_config.yml`

## Running tracker

```
cd application\main
python app_track.py
```

## Saving result
Results can be save to databse: `upload_db` in file `config.yml`
<p>
<img src="videos/db.PNG" width="500"/>
</p>

## FastAPI

```
cd application\main
uvicorn app_API:app --host 0.0.0.0 --port 8000 --reload

```
<p>
<img src="videos/fastapi.PNG" width="500"/>
</p>


