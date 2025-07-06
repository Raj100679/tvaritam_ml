# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('model.pkl', '.'), ('le_specials.pkl', '.'), ('le_mods.pkl', '.'), ('one_hot_columns.txt', '.')],
    hiddenimports=[
        'sklearn',
        'sklearn.pipeline',
        'sklearn.preprocessing',
        'sklearn.ensemble',
        'sklearn.tree',
        'sklearn.base',
        'sklearn.utils',
        'pandas',
        'joblib',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SalesPredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
