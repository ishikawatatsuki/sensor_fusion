from numpy import pi

DECLINATION_OFFSET_RADIAN_IN_ESTONIA = (9.98 + (1/60 * 59)) * (pi / 180)

MAX_CONSECUTIVE_DROPOUT_RATIO = 0.15

KITTI_DATE_MAPS = {
  "0027": "2011_10_03",
  "0042": "2011_10_03",
  "0034": "2011_10_03",
  "0067": "2011_09_26",
  "0016": "2011_09_30",
  "0018": "2011_09_30",
  "0020": "2011_09_30",
  "0027": "2011_09_30",
  "0028": "2011_09_30",
  "0033": "2011_09_30",
  "0034": "2011_09_30",
}

KITTI_SEQUENCE_MAPS = {
  "0027": "00",
  "0042": "01",
  "0034": "02",
  "0067": "03",
  "0016": "04",
  "0018": "05",
  "0020": "06",
  "0027": "07",
  "0028": "08",
  "0033": "09",
  "0034": "10",
}

VIZTRACK_VARIANT_MAPS = {
  "paldiski_01": "Paldiski/Exp1_var1",
  "vlaardingen_01": "Vlaardingen/trip_1_14_11_2024"
}