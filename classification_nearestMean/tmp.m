ids_nanAll = all(isnan(TS_DataMat), 1);
ids_nanSome = ids_nan & ~ids_nanAll;
sum(ids_nanAll)
sum(ids_nanSome)