{{- define "heart-disease-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-disease-api.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "heart-disease-api.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-disease-api.labels" -}}
app.kubernetes.io/name: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "heart-disease-api.selectorLabels" -}}
app: {{ include "heart-disease-api.name" . }}
{{- end -}}
