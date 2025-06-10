

targets = ['nt_core', 'nt_types', 'nt_functional', 'nt_functional_cpu',
            'nt_matmult', 'nt_linalg', 'nt_svd', 'nt_qr', 'nt_inv',
            'nt_column_space', 'nt_null_space', 'nt_ai', 'nt_tda',
            'nt_sparse', 'nt_images', 'nt_memory', 'nt_multi_processing', 'nt_fmri']


def print_path(target):
    print("""
      - name: Restore {} build cache
        uses: actions/cache@v4
        with:
          path: temp/{}/
          key: neurotensor-win-build-{}-""".format(target, target, target), end= '')
    print("${{ github.run_id }}", end = '')
    print("""
          restore-keys: neurotensor-win-build-{}-
          """.format(target))


for target in targets:
    print_path(target)
