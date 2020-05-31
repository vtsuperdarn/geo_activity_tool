import matplotlib


def create_custom_cmap():

    col_arr = ["#fee7e9",
            "#fee7e7",
            "#fee7e5",
            "#fee6e2",
            "#fee6e0",
            "#fee6de",
            "#fee6db",
            "#fee6d9",
            "#fee5d7",
            "#fee5d5",
            "#fee5d2",
            "#fee5d0",
            "#fee4ce",
            "#fee4cb",
            "#fee4c9",
            "#fee4c7",
            "#fee4c4",
            "#fee3c2",
            "#fee3c0",
            "#fee3bd",
            "#fee3bb",
            "#fee2b9",
            "#fee2b6",
            "#ffe2b4",
            "#ffe2b2",
            "#ffe2af",
            "#ffe1ad",
            "#ffe1ab",
            "#ffe1a9",
            "#ffe1a6",
            "#ffe1a4",
            "#ffe0a2",
            "#ffe09f",
            "#ffe09d",
            "#ffe09b",
            "#ffdf98",
            "#ffdf96",
            "#ffdf94",
            "#ffdf91",
            "#ffdf8f",
            "#ffde8d",
            "#ffde8a",
            "#ffde88",
            "#ffde86",
            "#ffdd83",
            "#ffdd81",
            "#ffdd7f",
            "#ffdd7d",
            "#ffdd7a",
            "#ffdc78",
            "#ffdc76",
            "#ffdc73",
            "#ffdc71",
            "#ffdb6f",
            "#ffdb6c",
            "#ffdb6a",
            "#ffdb68",
            "#ffdb65",
            "#ffda63",
            "#ffda61",
            "#ffda5e",
            "#ffda5c",
            "#ffda5a",
            "#ffd958",
            "#ffd956",
            "#ffd855",
            "#ffd755",
            "#ffd654",
            "#ffd554",
            "#ffd454",
            "#ffd353",
            "#ffd253",
            "#ffd152",
            "#ffd052",
            "#ffcf51",
            "#ffce51",
            "#ffcd51",
            "#ffcc50",
            "#ffcb50",
            "#ffca4f",
            "#ffc94f",
            "#ffc84e",
            "#ffc74e",
            "#ffc64e",
            "#ffc54d",
            "#ffc44d",
            "#ffc34c",
            "#ffc24c",
            "#ffc14b",
            "#ffc04b",
            "#ffbf4b",
            "#ffbe4a",
            "#ffbd4a",
            "#ffbc49",
            "#ffbb49",
            "#ffba49",
            "#ffb948",
            "#ffb848",
            "#ffb747",
            "#ffb647",
            "#ffb546",
            "#ffb446",
            "#ffb346",
            "#ffb245",
            "#ffb145",
            "#ffb044",
            "#ffaf44",
            "#ffae43",
            "#ffad43",
            "#ffac43",
            "#ffab42",
            "#ffaa42",
            "#ffa941",
            "#ffa841",
            "#ffa740",
            "#ffa640",
            "#ffa540",
            "#ffa43f",
            "#ffa33f",
            "#ffa23e",
            "#ffa13e",
            "#ffa03d",
            "#ff9f3d",
            "#ff9e3d",
            "#ff9d3c",
            "#ff9c3c",
            "#ff9b3b",
            "#ff9a3b",
            "#ff9a3b",
            "#ff993b",
            "#ff993b",
            "#ff983c",
            "#ff983c",
            "#ff973c",
            "#ff973c",
            "#ff963d",
            "#ff963d",
            "#ff963d",
            "#ff953d",
            "#ff953e",
            "#ff943e",
            "#ff943e",
            "#ff933e",
            "#ff933f",
            "#ff923f",
            "#ff923f",
            "#ff913f",
            "#ff9140",
            "#ff9140",
            "#ff9040",
            "#ff9040",
            "#ff8f41",
            "#ff8f41",
            "#ff8e41",
            "#ff8e41",
            "#ff8d42",
            "#ff8d42",
            "#ff8d42",
            "#ff8c42",
            "#ff8c43",
            "#ff8b43",
            "#ff8b43",
            "#ff8a44",
            "#ff8a44",
            "#ff8944",
            "#ff8944",
            "#ff8945",
            "#ff8845",
            "#ff8845",
            "#ff8745",
            "#ff8746",
            "#ff8646",
            "#ff8646",
            "#ff8546",
            "#ff8547",
            "#ff8447",
            "#ff8447",
            "#ff8447",
            "#ff8348",
            "#ff8348",
            "#ff8248",
            "#ff8248",
            "#ff8149",
            "#ff8149",
            "#ff8049",
            "#ff8049",
            "#ff804a",
            "#ff7f4a",
            "#ff7f4a",
            "#ff7e4a",
            "#ff7e4b",
            "#ff7d4b",
            "#ff7d4a",
            "#ff7c49",
            "#ff7b48",
            "#ff7a48",
            "#ff7947",
            "#ff7946",
            "#ff7845",
            "#ff7744",
            "#ff7643",
            "#ff7642",
            "#ff7541",
            "#ff7440",
            "#ff733f",
            "#ff733e",
            "#ff723e",
            "#ff713d",
            "#ff703c",
            "#ff6f3b",
            "#ff6f3a",
            "#ff6e39",
            "#ff6d38",
            "#ff6c37",
            "#ff6c36",
            "#ff6b35",
            "#ff6a34",
            "#ff6934",
            "#ff6833",
            "#ff6832",
            "#ff6731",
            "#ff6630",
            "#ff652f",
            "#ff652e",
            "#ff642d",
            "#ff632c",
            "#ff622b",
            "#ff612a",
            "#ff612a",
            "#ff6029",
            "#ff5f28",
            "#ff5e27",
            "#ff5e26",
            "#ff5d25",
            "#ff5c24",
            "#ff5b23",
            "#ff5a22",
            "#ff5a21",
            "#ff5920",
            "#ff5820",
            "#ff571f",
            "#ff571e",
            "#ff561d",
            "#ff551c",
            "#ff541b",
            "#ff531a",
            "#ff5319",
            "#ff5218",
            "#ff5117",
            "#ff5016",
            "#ff5016",
            "#ff4f15",
            "#ff4e14",
            "#ff4d13",
            "#ff4d12",
            "#ff4c11"]

    return matplotlib.colors.ListedColormap(col_arr,name="my_reds")