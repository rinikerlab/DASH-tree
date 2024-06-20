from serenityff.charge.tree.tree_factory import Forest


def test_MBIS_DASH_tree() -> None:
    tree = Forest.get_MBIS_DASH_tree(preload=False)
    tree.default_std_column = "std"
    tree.default_value_column = "result"
