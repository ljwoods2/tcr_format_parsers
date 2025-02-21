from pathlib import Path
import polars as pl
import h5py
import numpy as np
import re
import scipy.sparse as sps


def use_32bit_dtypes(df):
    schema = df.collect_schema()
    # ensure we use 32 bit datatypes
    type_mapping = {
        pl.Int64: pl.Int32,
        pl.Float64: pl.Float32,
    }
    df = df.with_columns(
        [
            pl.col(col_name).cast(type_mapping[col_dtype], strict=False)
            for col_name, col_dtype in schema.items()
            if col_dtype in type_mapping
        ]
    )
    return df


class CellRangerOutput:

    def __init__(self, directory_path: list):

        dir_path = None
        if isinstance(directory_path, str):
            dir_path = Path(directory_path)
        elif isinstance(directory_path, Path):
            dir_path = directory_path
        else:
            raise ValueError

        self.dir_path = dir_path
        self.job_name = dir_path.name
        self.donor_num = re.findall(r"\d+", self.job_name)[0]

    def get_filtered_contig_df(self):
        """
        Returns CellRangers' vdj pipeline `filtered_contig_annotations.csv` as a polars LazyFrame.

        https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-5p-outputs-annotations-vdj

        Returns
        -------
        pl.LazyFrame
        """
        filt_annot_path = self.dir_path / Path(
            f"outs/per_sample_outs/{self.job_name}/vdj_t/filtered_contig_annotations.csv"
        )

        filt_annot_df = pl.scan_csv(filt_annot_path)
        filt_annot_df = use_32bit_dtypes(filt_annot_df)
        return filt_annot_df

    def get_feature_matrix_barcodes_as_df(self):
        """
        https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-h5-matrices

        Returns
        -------
        pl.LazyFrame
            barcode : str
                cell barcode
            index : u32
                index of barcode in feature matrix
        """
        mtx_h5 = h5py.File(
            self.dir_path / "outs/multi/count/raw_feature_bc_matrix.h5"
        )
        barcode_ndarr = mtx_h5["matrix"]["barcodes"][:].astype(str)
        return pl.DataFrame({"barcode": barcode_ndarr}).with_row_index()

    def get_feature_matrix_featnames_as_df(self):
        """
        https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-h5-matrices

        Returns
        -------
        pl.LazyFrame
            feature_name : str
                name of feature
            index : u32
                index of feature in feature matrix
        """
        mtx_h5 = h5py.File(
            self.dir_path / "outs/multi/count/raw_feature_bc_matrix.h5"
        )
        featname_ndarr = mtx_h5["matrix"]["features"]["name"][:].astype(str)
        return pl.DataFrame({"feature_name": featname_ndarr}).with_row_index()

    def get_feature_matrix_as_ndarr(self):
        """
        https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-h5-matrices

        Returns
        -------
        mtx_ndarr : np.ndarray[int32]
            feature barcode matrix
        feature_name_ndarr : np.ndarray[str]
            feature names
        barcode_ndarr : np.ndarray[str]
            cell barcodes
        """
        mtx_h5 = h5py.File(
            self.dir_path / "outs/multi/count/raw_feature_bc_matrix.h5"
        )
        barcode_ndarr = mtx_h5["matrix"]["barcodes"][:].astype(str)
        feature_name_ndarr = mtx_h5["matrix"]["features"]["name"][:].astype(
            str
        )
        mtx_ndarr = sps.csc_matrix(
            (
                mtx_h5["matrix"]["data"][:],
                mtx_h5["matrix"]["indices"][:],
                mtx_h5["matrix"]["indptr"][:],
            ),
            shape=mtx_h5["matrix"]["shape"][:],
            dtype=np.int32,
        ).toarray()

        return (mtx_ndarr, feature_name_ndarr, barcode_ndarr)

    def get_feature_matrix_as_df(self):
        barcode_ndarr, feature_name_ndarr, mtx_ndarr = (
            self.get_feature_matrix_as_ndarr()
        )
        tmp_dfs = []

        for i, feature_name in enumerate(feature_name_ndarr[:]):
            tmp_df = pl.DataFrame(
                {
                    "feature_name": feature_name,
                    "barcode": barcode_ndarr[:],
                    "count": mtx_ndarr[i, :],
                }
            )
            tmp_dfs.append(tmp_df)

        mtx_df = pl.concat(tmp_dfs, how="vertical")

        return mtx_df

    def get_featbcmatrix_obj(self, name):
        mtx, _, _ = self.get_feature_matrix_as_ndarr()
        bc_df = self.get_feature_matrix_barcodes_as_df()
        featname_df = self.get_feature_matrix_featnames_as_df()
        return FeatBCMatrix(name, mtx, featname_df, bc_df)


class FeatBCMatrix:

    def __init__(self, name, mtx, featnames_idx_df, bc_idx_df):
        self.parent = None
        self.mtx = mtx
        self.name = name
        self.idx_name = name + "_idx"

        if "index" in featnames_idx_df.columns:
            rename = {"index": self.idx_name}
        else:
            rename = {}

        self.featnames_idx_df = featnames_idx_df.rename(rename).sort(
            by=self.idx_name
        )
        self.bc_idx_df = bc_idx_df.rename(rename).sort(by=self.idx_name)

    def get_featnames_ndarr(self):
        return (
            self.featnames_idx_df.select("feature_name").to_series().to_numpy()
        )

    def get_bc_ndarr(self):
        return self.bc_idx_df.select("barcode").to_series().to_numpy()

    def get_ndarr_for_col(
        self,
        df,
        col_name,
    ):
        if df is None:
            raise ValueError(
                "Must specify either 'fbc' or 'bc' for dataframe argument"
            )
        else:
            df = self.featnames_idx_df if df == "fbc" else self.bc_idx_df

        return df.select(col_name).to_series().to_numpy()

    def create_child_matrix(
        self,
        name,
        featnames_df=None,
        bc_df=None,
        featname_cols=None,
        bc_cols=None,
    ):
        new_idx_name = name + "_idx"
        if new_idx_name == self.idx_name:
            raise ValueError("Child matrix name cannot be the same as parent")

        feature_slice = slice(None)
        bc_slice = slice(None)

        if featnames_df is not None:
            if featname_cols is not None:
                selections = [col for col in featname_cols]
            else:
                selections = []

            new_featnames_idx_df = (
                self.featnames_idx_df.join(
                    featnames_df.select(["feature_name"] + selections),
                    on="feature_name",
                    how="inner",
                )
                .unique()
                .sort(by=self.idx_name)
                .with_row_index(name=new_idx_name)
            )
            # slice based on the subselection
            feature_slice = new_featnames_idx_df.select(
                self.idx_name
            ).to_series()
        else:
            new_featnames_idx_df = self.featnames_idx_df.with_row_index(
                name=new_idx_name
            )

        if bc_df is not None:
            if bc_cols is not None:
                selections = [col for col in bc_cols]
            else:
                selections = []

            new_bc_idx_df = (
                self.bc_idx_df.join(
                    bc_df.select(["barcode"] + selections),
                    on="barcode",
                    how="inner",
                )
                .unique()
                .sort(by=self.idx_name)
                .with_row_index(name=new_idx_name)
            )
            bc_slice = new_bc_idx_df.select(self.idx_name).to_series()
        else:
            new_bc_idx_df = self.bc_idx_df.with_row_index(name=new_idx_name)

        mtx = self.mtx[feature_slice][:, bc_slice]

        return FeatBCMatrix(name, mtx, new_featnames_idx_df, new_bc_idx_df)
