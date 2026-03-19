<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > Resources > Useful Software
        </div>
        <div class="aoi title">
            <h2>Useful Software</h2>
        </div>
        <div class="aoi content">
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>NoteExpress</td>
                        <td>Reference discovery and management software for searching, downloading, organizing, and
                            citing literature. It integrates with Microsoft Word to format references automatically.
                        </td>
                    </tr>
                    <tr>
                        <td>EndNote</td>
                        <td>A reference management tool published by Clarivate. University users can access the online
                            version through the Web of Knowledge platform.</td>
                    </tr>
                    <tr>
                        <td>Adobe Acrobat Reader</td>
                        <td>A standard reader for PDF documents.</td>
                    </tr>
                    <tr>
                        <td>CAJViewer</td>
                        <td>Supports multiple document formats including CAJ, NH, KDH, TEB, and PDF.</td>
                    </tr>
                    <tr>
                        <td>Chaoxing Reader</td>
                        <td>Used to view or download Chaoxing e-books.</td>
                    </tr>
                    <tr>
                        <td>Campus-Licensed Software</td>
                        <td>Includes licensed copies of Windows, Office, Visio, WPS, MATLAB, and other software
                            packages. Download is available through the university intranet portal.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</template>

<script setup>
</script>

<style scoped>
.title {
    text-align: center;
}

.main-content {
    padding: 20px 5%;
}

.content {
    text-indent: 2em;
    font-size: 18px;
    line-height: 1.6;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    /* Add vertical spacing */
}

th,
td {
    border: 1px solid #ccc;
    padding: 10px;
    text-align: left;
}

th {
    background-color: #f7f7f7;
}
</style>