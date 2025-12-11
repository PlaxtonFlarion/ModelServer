import modal

from apps.cross_enc   import app as cross_enc_app
from apps.embed_en    import app as embed_en_app
from apps.embed_zh    import app as embed_zh_app
from apps.infer_color import app as infer_color_app
from apps.infer_faint import app as infer_faint_app

from utils import const

app = modal.App(const.GROUP_FUNC)

app.include(cross_enc_app    )
app.include(embed_en_app     )
app.include(embed_zh_app     )
app.include(infer_color_app  )
app.include(infer_faint_app  )


if __name__ == '__main__':
    pass
