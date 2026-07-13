{{- define "travel-time.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "travel-time.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "travel-time.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "travel-time.labels" -}}
app.kubernetes.io/name: {{ include "travel-time.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "travel-time.selectorLabels" -}}
app.kubernetes.io/name: {{ include "travel-time.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
